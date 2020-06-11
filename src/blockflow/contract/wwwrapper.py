from typing import Optional, Union, Type, NamedTuple, Dict
from datetime import timedelta, datetime
from functools import lru_cache
import logging

import web3
import web3.contract
import web3.types
import web3.exceptions
from eth_keys import keys, KeyAPI
from eth_typing.evm import BlockNumber
import coincurve
from hexbytes.main import HexBytes
from eth_utils.conversions import to_int
from eth_account._utils.signing import extract_chain_id, to_standard_v
from eth_account._utils.transactions import ALLOWED_TRANSACTION_KEYS, serializable_unsigned_transaction_from_dict

from .ethereum_address import EthereumAddress
from .._utils.log_wrapper import log
from .._utils.threadsafe_counter import get_global_counter

class VyperOutput(NamedTuple):
    bytecode: object
    abi: object
    src_map: object

_ADDRESS_TO_NONCE_TO_TXN_HASH: Dict[EthereumAddress, Dict[web3.types.Nonce, HexBytes]] = {}

class WWWrapper:
    _logger = logging.getLogger(__name__)

    def __init__(self, private_key: bytes, w3: web3.Web3) -> None:
        self._private_key = private_key
        self._public_key = keys.PrivateKey(private_key).public_key
        self._address: EthereumAddress = self._public_key.to_checksum_address()

        self._nonce_counter = get_global_counter(self._address, w3.eth.getTransactionCount(self._address))
        with self._nonce_counter:
            if self._address not in _ADDRESS_TO_NONCE_TO_TXN_HASH:
                _ADDRESS_TO_NONCE_TO_TXN_HASH[self._address] = {}
        self._nonce_to_txn_hash = _ADDRESS_TO_NONCE_TO_TXN_HASH[self._address]

        assert w3.isConnected(), "couldn't connect to ethereum"
        self._w3 = w3
        self._chain_id = w3.eth.chainId
        # self._gasPrice = w3.eth.gasPrice
        # self._gas_limit = w3.eth.getBlock('latest')['gasLimit']
        self._logger.info("connected to web3")

    @log(_logger, logging.DEBUG)
    def get_balance(self) -> web3.types.Wei:
        return self._w3.eth.getBalance(self._address)

    @log(_logger, logging.DEBUG)
    def get_current_nonce(self) -> web3.types.Nonce:
        return web3.types.Nonce(self._nonce_counter.get_value())

    @log(_logger, logging.DEBUG)
    def get_txn_hash(self, nonce: web3.types.Nonce) -> HexBytes:
        return self._nonce_to_txn_hash[nonce]

    @log(_logger, logging.DEBUG, log_args=False)
    def build_contract_type(self, vyper_output: VyperOutput) -> Type[web3.contract.Contract]:
        return self._w3.eth.contract(bytecode=vyper_output.bytecode, abi=vyper_output.abi, src_map=vyper_output.src_map)

    @property
    def address(self) -> EthereumAddress:
        return self._address

    @log(_logger, logging.DEBUG)
    def _get_public_key(self, transaction_digest: HexBytes) -> KeyAPI.PublicKey:
        tx_data: web3.types.TxData = self._w3.eth.getTransaction(transaction_digest)
        # adapted from https://ethereum.stackexchange.com/questions/2166/retrieve-the-signature-of-a-transaction-on-the-blockchain
        chain_id, v_bit = extract_chain_id(tx_data.v)
        signature = KeyAPI.Signature(vrs=(to_standard_v(v_bit), to_int(tx_data.r), to_int(tx_data.s)))
        new_tx = {key: tx_data[key] for key in ALLOWED_TRANSACTION_KEYS - {'chainId', 'data'}}
        new_tx['data'] = tx_data.input
        new_tx['chainId'] = chain_id
        unsigned_tx = serializable_unsigned_transaction_from_dict(new_tx)
        public_key = signature.recover_public_key_from_msg_hash(unsigned_tx.hash())
        self._logger.debug("recovered address: %s", public_key.to_address())
        return public_key

    @log(_logger, logging.DEBUG)
    def derive_ecdh_key(self, transaction_digest: HexBytes) -> bytes:
        client_public_key_eth = self._get_public_key(transaction_digest)
        client_public_key_compressed_bytes = client_public_key_eth.to_compressed_bytes()
        my_private_key_coincurve = coincurve.PrivateKey(self._private_key)
        shared_secret = my_private_key_coincurve.ecdh(client_public_key_compressed_bytes)
        assert isinstance(shared_secret, bytes)
        return shared_secret

    @log(_logger, logging.DEBUG)
    def get_latest_block_number(self) -> BlockNumber:
        latest_block = self._w3.eth.getBlock('latest')
        latest_block_number = BlockNumber(latest_block['number'])
        self._logger.debug("Latest block number: %d", latest_block_number)
        return latest_block_number

    @log(_logger, logging.DEBUG)
    @lru_cache()
    def get_average_block_interval(self) -> timedelta:
        latest_block = self._w3.eth.getBlock('latest')
        latest_block_number = latest_block['number']
        older_block_number = max(0, latest_block_number - 1)
        assert older_block_number >= 0, f"invalid block: {older_block_number}"
        num_blocks = latest_block_number - older_block_number
        older_block = self._w3.eth.getBlock(BlockNumber(older_block_number))
        latest_block_timestamp = datetime.fromtimestamp(latest_block['timestamp'])
        older_block_timestamp = datetime.fromtimestamp(older_block['timestamp'])
        block_interval = (latest_block_timestamp - older_block_timestamp) / num_blocks
        assert isinstance(block_interval, timedelta)
        return block_interval

    @log(_logger, logging.DEBUG)
    def estimate_time_of_block(self, block_number: BlockNumber) -> datetime:
        latest_block = self._w3.eth.getBlock('latest')
        latest_block_number = latest_block['number']
        if latest_block_number == block_number:
            return datetime.fromtimestamp(latest_block['timestamp'])
        if block_number < latest_block_number:
            historical_block = self._w3.eth.getBlock(block_number)
            return datetime.fromtimestamp(historical_block['timestamp'])
        diff = block_number - latest_block_number
        assert diff >= 0, "diff is negative"
        block_interval = self.get_average_block_interval()
        block_timedelta = block_interval * diff
        latest_block_timestamp = datetime.fromtimestamp(latest_block['timestamp'])
        estimated_time = latest_block_timestamp + block_timedelta
        assert isinstance(estimated_time, datetime)
        self._logger.debug("Estimating time of block %d to be %s", block_number, estimated_time)
        return estimated_time

    @log(_logger, logging.DEBUG)
    def estimate_block_at_time(self, timestamp: datetime) -> BlockNumber:
        latest_block = self._w3.eth.getBlock('latest')
        time_diff = timestamp - datetime.fromtimestamp(latest_block['timestamp'])
        block_interval = self.get_average_block_interval()
        return BlockNumber(latest_block['number'] + int(time_diff/block_interval))

    @log(_logger, logging.INFO)
    def transact(self, contract_function: Union[web3.contract.ContractFunction, web3.contract.ContractConstructor], tx_params: Optional[web3.types.TxParams] = None) -> HexBytes:
        if tx_params is None:
            tx_params = {}
        tx_params['from'] = self._address
        # tx_params['gas'] = self._gas_limit  # no harm in setting the gas limit too high? save an rpc
        tx_params['chainId'] = self._chain_id
        # tx_params['gasPrice'] = self._gasPrice
        with self._nonce_counter:
            nonce = web3.types.Nonce(self._nonce_counter.get_value())
            tx_params['nonce'] = nonce
            tx_params = contract_function.buildTransaction(tx_params)
            signed_tx = self._w3.eth.account.sign_transaction(tx_params, self._private_key)
            txn_hash = self._w3.eth.sendRawTransaction(signed_tx.rawTransaction)
            self._nonce_counter.increment() # only increase the nonce if we are able to successfully send the transaction
            self._nonce_to_txn_hash[nonce] = txn_hash
            assert isinstance(txn_hash, HexBytes)
            return txn_hash

    @log(_logger, logging.DEBUG)
    def get_transaction_receipt(self, txn_hash: HexBytes) -> web3.types.TxReceipt:
        tx_receipt: web3.types.TxReceipt = self._w3.eth.getTransactionReceipt(txn_hash)
        address: Optional[EthereumAddress] = tx_receipt.contractAddress
        if address is None:
            address = tx_receipt.to
        self._logger.info("Mined transaction %s at address %s. Consumed %d gas", txn_hash.hex(), address, tx_receipt.gasUsed)
        return tx_receipt
