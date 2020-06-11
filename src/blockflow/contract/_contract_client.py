from typing import Sequence, Optional, List, Mapping, Dict, Set, Tuple, NamedTuple, cast, Iterable
from fractions import Fraction
import logging
from functools import lru_cache
from datetime import datetime, timedelta
import time

from cached_property import cached_property
from hexbytes.main import HexBytes
import cid
import web3
import web3.contract
import web3.types
import web3.exceptions
from eth_utils.crypto import keccak
from eth_utils.encoding import int_to_big_endian
from eth_typing.evm import BlockNumber

from .ethereum_address import EthereumAddress
from .contract_parameters import ContractParameters
from ._contract_wwwrapper_pool import ContractWWWrapperPool
from .._utils.log_wrapper import log

class DecryptionKeyResponse(NamedTuple):
    dp_round: int
    auditor: EthereumAddress
    client: EthereumAddress
    provider: EthereumAddress
    decryption_key_ipfs_address: cid.CIDv0

class EventNotFoundError(Exception):
    pass

class RevertException(Exception):
    pass

class ContractClient:
    _logger = logging.getLogger(__name__)

    def __init__(self, pool: ContractWWWrapperPool):
        self._pool = pool

    @cached_property
    def address(self) -> EthereumAddress:
        with self._pool.get() as (dummy_contract, wwwrapper):
            return wwwrapper.address

    @cached_property
    def parameters(self) -> ContractParameters:
        return ContractParameters(
            num_dp_rounds=self.num_dp_rounds,
            bond_amount=self.bond_amount,
            bond_reserve_amount=self.bond_reserve_amount,
            start_block=self.start_block,
            dp_round_training_blockdelta=self._dp_round_training_blockdelta,
            dp_round_data_retrieval_blockdelta=self._dp_round_data_retrieval_blockdelta,
            dp_round_scoring_blockdelta=self._dp_round_scoring_blockdelta,
            dp_round_score_decrypting_blockdelta=self._dp_round_score_decrypting_blockdelta,
            min_agreement_threshold_num=self.min_agreement_threshold.numerator,
            min_agreement_threshold_denom=self.min_agreement_threshold.denominator,
            refund_fraction_num=self.refund_fraction.numerator,
            refund_fraction_denom=self.refund_fraction.denominator,
            submission_blockdelta=self._submission_blockdelta,
            client_authorizer=str(self.client_authorizer)
        )

    @cached_property
    def _submission_blockdelta(self) -> int:
        with self._pool.get() as (contract, dummy_wwwrapper):
            answer = contract.functions.get_submission_blockdelta().call()
            assert isinstance(answer, int)
            return answer

    @cached_property
    def enrollment_start_block(self) -> BlockNumber:
        return BlockNumber(max(0, self.start_block - self._submission_blockdelta))

    @lru_cache()
    def get_training_submission_start_block(self, dp_round: int) -> BlockNumber:
        return BlockNumber(max(self.get_data_retrieval_start_block(dp_round) - self._submission_blockdelta, self.get_dp_round_start_block(dp_round)))

    def get_balance(self) -> web3.types.Wei:
        with self._pool.get() as (dummy_contract, wwwrapper):
            return wwwrapper.get_balance()

    @lru_cache()
    def get_data_retrieval_start_block(self, dp_round: int) -> BlockNumber:
        return BlockNumber(self.get_dp_round_start_block(dp_round) + self._dp_round_training_blockdelta)

    @lru_cache()
    def get_data_retrieval_submission_start_block(self, dp_round: int) -> BlockNumber:
        return BlockNumber(max(self.get_scoring_start_block(dp_round) - self._submission_blockdelta, self.get_data_retrieval_start_block(dp_round)))

    @lru_cache()
    def get_scoring_start_block(self, dp_round: int) -> BlockNumber:
        return BlockNumber(self.get_data_retrieval_start_block(dp_round) + self._dp_round_data_retrieval_blockdelta)

    @lru_cache()
    def get_scoring_submission_start_block(self, dp_round: int) -> BlockNumber:
        return BlockNumber(max(self.get_score_decrypting_start_block(dp_round) - self._submission_blockdelta, self.get_scoring_start_block(dp_round)))

    @lru_cache()
    def get_score_decrypting_start_block(self, dp_round: int) -> BlockNumber:
        return BlockNumber(self.get_scoring_start_block(dp_round) + self._dp_round_scoring_blockdelta)

    @lru_cache()
    def get_score_decrypting_submission_start_block(self, dp_round: int) -> BlockNumber:
        return BlockNumber(max(self.get_dp_round_start_block(dp_round + 1) - self._submission_blockdelta, self.get_score_decrypting_start_block(dp_round)))

    @log(_logger, logging.DEBUG)
    def get_latest_block_number(self) -> BlockNumber:
        with self._pool.get() as (dummy_contract, wwwrapper):
            return wwwrapper.get_latest_block_number()

    @log(_logger, logging.DEBUG)
    def get_average_block_interval(self, lookback_period: int) -> timedelta:
        with self._pool.get() as (dummy_contract, wwwrapper):
            answer = wwwrapper.get_average_block_interval(lookback_period)
            assert isinstance(answer, timedelta)
            return answer

    @log(_logger, logging.DEBUG)
    def estimate_time_of_block(self, block_number: BlockNumber) -> datetime:
        with self._pool.get() as (dummy_contract, wwwrapper):
            return wwwrapper.estimate_time_of_block(block_number)

    @log(_logger, logging.DEBUG)
    def estimate_block_at_time(self, timestamp: datetime) -> BlockNumber:
        with self._pool.get() as (dummy_contract, wwwrapper):
            return wwwrapper.estimate_block_at_time(timestamp)

    @log(_logger, logging.INFO, log_return=False)
    def wait_for_block(self, block_number: BlockNumber) -> None:
        latest_block_number = self.get_latest_block_number()
        latest_block_timestamp = self.estimate_time_of_block(latest_block_number)
        if block_number <= latest_block_number:
            self._logger.warning("Block %d is in the past (currently at %d); returning immediately", block_number, latest_block_number)
            return
        while latest_block_number < block_number:
            diff = block_number - latest_block_number
            older_block_number = max(1, latest_block_number - diff)
            num_blocks = latest_block_number - older_block_number
            older_block_timestamp = self.estimate_time_of_block(BlockNumber(older_block_number))
            block_rate = (latest_block_timestamp - older_block_timestamp) / num_blocks
            block_timedelta = block_rate * diff
            sleep_time = max(1.0, block_timedelta.total_seconds())
            self._logger.info("Sleeping %f secs (%d blocks) for block %d (interval = 1 block/%f seconds). Currently at block %d", sleep_time, diff, block_number, block_rate.total_seconds(), latest_block_number)
            time.sleep(sleep_time)
            latest_block_number = self.get_latest_block_number()
            latest_block_timestamp = self.estimate_time_of_block(latest_block_number)

    @log(_logger, logging.INFO)
    def collect_reward(self, dp_round: int) -> HexBytes:
        with self._pool.get() as (contract, wwwrapper):
            return wwwrapper.transact(contract.functions.collect_reward(dp_round))

    @log(_logger, logging.INFO, log_return=False)
    @lru_cache()
    def wait_for_tx_receipt(self, txn_hash: HexBytes, deadline: Optional[datetime] = None) -> web3.types.TxReceipt:
        expiration = None if deadline is None else deadline
        while True:
            try:
                with self._pool.get() as (dummy_contract, wwwrapper):
                    tx_receipt = wwwrapper.get_transaction_receipt(txn_hash)
                return tx_receipt
            except web3.exceptions.TransactionNotFound:
                if expiration is None or datetime.now() < expiration:
                    time.sleep(1.1)
                else:
                    raise

    @log(_logger, logging.INFO, log_return=False)
    def validate_tx_receipt(self, tx_receipt: web3.types.TxReceipt) -> web3.types.TxReceipt:
        if not tx_receipt['status']:
            self._logger.error("Transaction %s was reverted", tx_receipt['transactionHash'])
            raise RevertException(f"Transaction {tx_receipt['transactionHash']} was reverted")

    @log(_logger, logging.INFO, log_return=False)
    def get_txn_hash(self, nonce: web3.types.Nonce) -> HexBytes:
        with self._pool.get() as (dummy_contract, wwwrapper):
            return wwwrapper.get_txn_hash(nonce)

    @log(_logger, logging.INFO, log_return=False)
    def get_current_nonce(self) -> web3.types.Nonce:
        with self._pool.get() as (dummy_contract, wwwrapper):
            return wwwrapper.get_current_nonce()

    @cached_property
    def score_denom(self) -> int:
        with self._pool.get() as (contract, dummy_wwwrapper):
            answer = contract.functions.get_score_denom().call()
            assert isinstance(answer, int)
            return answer

    @cached_property
    def min_agreement_threshold(self) -> Fraction:
        with self._pool.get() as (contract, dummy_wwwrapper):
            num, denom = contract.functions.get_min_agreement_threshold().call()
            assert isinstance(num, int)
            assert isinstance(denom, int)
            return Fraction(num, denom)

    @cached_property
    def bond_amount(self) -> int:
        with self._pool.get() as (contract, dummy_wwwrapper):
            answer = contract.functions.get_bond_amount().call()
            assert isinstance(answer, int)
            return answer

    @cached_property
    def bond_reserve_amount(self) -> int:
        with self._pool.get() as (contract, dummy_wwwrapper):
            answer = contract.functions.get_bond_reserve_amount().call()
            assert isinstance(answer, int)
            return answer

    @cached_property
    def refund_fraction(self) -> Fraction:
        with self._pool.get() as (contract, dummy_wwwrapper):
            num, denom = contract.functions.get_refund_fraction().call()
            assert isinstance(num, int)
            assert isinstance(denom, int)
            return Fraction(num, denom)

    @cached_property
    def client_authorizer(self) -> EthereumAddress:
        with self._pool.get() as (contract, dummy_wwwrapper):
            return EthereumAddress(contract.functions.get_client_authorizer().call())

    @cached_property
    def start_block(self) -> BlockNumber:
        with self._pool.get() as (contract, dummy_wwwrapper):
            return BlockNumber(contract.functions.get_start_block().call())

    @cached_property
    def _dp_round_training_blockdelta(self) -> int:
        with self._pool.get() as (contract, dummy_wwwrapper):
            answer = contract.functions.get_dp_round_training_blockdelta().call()
            assert isinstance(answer, int)
            return answer

    @cached_property
    def _dp_round_data_retrieval_blockdelta(self) -> int:
        with self._pool.get() as (contract, dummy_wwwrapper):
            answer = contract.functions.get_dp_round_data_retrieval_blockdelta().call()
            assert isinstance(answer, int)
            return answer

    @cached_property
    def _dp_round_scoring_blockdelta(self) -> int:
        with self._pool.get() as (contract, dummy_wwwrapper):
            answer = contract.functions.get_dp_round_scoring_blockdelta().call()
            assert isinstance(answer, int)
            return answer

    @cached_property
    def _dp_round_score_decrypting_blockdelta(self) -> int:
        with self._pool.get() as (contract, dummy_wwwrapper):
            answer = contract.functions.get_dp_round_score_decrypting_blockdelta().call()
            assert isinstance(answer, int)
            return answer

    @cached_property
    def num_dp_rounds(self) -> int:
        with self._pool.get() as (contract, dummy_wwwrapper):
            answer = contract.functions.get_num_dp_rounds().call()
            assert isinstance(answer, int)
            return answer

    @cached_property
    def enrolled_clients(self) -> Sequence[EthereumAddress]:
        assert self.get_latest_block_number() >= self.start_block, "please wait until the experiment begins before fetching the list of clients"
        from_block = max(0, self.start_block - self._submission_blockdelta)
        to_block = self.start_block - 1
        with self._pool.get() as (contract, dummy_wwwrapper):
            events = cast(Iterable[web3.types.EventData], contract.events.ClientEnrolled.getLogs(fromBlock=from_block, toBlock=to_block))
        clients: List[EthereumAddress] = []
        for event in events:
            client = event['args']['client']
            clients.append(client)
        return clients

    @log(_logger, logging.DEBUG)
    @lru_cache()
    def get_dp_round(self, block: int) -> int:
        if block < self.start_block:
            return 0
        elapsed_blockdelta = block - self.start_block
        denom = self._dp_round_training_blockdelta + self._dp_round_data_retrieval_blockdelta + self._dp_round_scoring_blockdelta + self._dp_round_score_decrypting_blockdelta
        round_without_offset = elapsed_blockdelta // denom
        round_with_offset = round_without_offset + 1
        dp_round = round_with_offset
        if __debug__:
            with self._pool.get() as (contract, dummy_wwwrapper):
                assert dp_round == contract.functions.get_dp_round(block).call()
        assert isinstance(dp_round, int)
        return dp_round

    @log(_logger, logging.INFO)
    def request_decryption_key(self, dp_round: int, client: EthereumAddress) -> HexBytes:
        with self._pool.get() as (contract, wwwrapper):
            return wwwrapper.transact(contract.functions.request_decryption_key(dp_round, client))

    @log(_logger, logging.INFO)
    def get_decryption_key_requests(self, dp_round: int, client: EthereumAddress) -> Sequence[Tuple[EthereumAddress, EthereumAddress]]:
        from_block = self.get_data_retrieval_start_block(dp_round)
        to_block = self.get_score_decrypting_start_block(dp_round) - 1
        with self._pool.get() as (contract, dummy_wwwrapper):
            events = cast(Iterable[web3.types.EventData], contract.events.DecryptionKeyRequest.getLogs({"_client": client, "_dp_round": dp_round}, fromBlock=from_block, toBlock=to_block))
        answer: List[Tuple[EthereumAddress, EthereumAddress]] = []
        for event in events:
            answer.append((event['args']['auditor'], event['args']['client']))
        return answer

    @log(_logger, logging.INFO)
    def provide_decryption_key(self, dp_round: int, auditor: EthereumAddress, client: EthereumAddress, key_ipfs_address: cid.CIDv0) -> HexBytes:
        with self._pool.get() as (contract, wwwrapper):
            return wwwrapper.transact(contract.functions.provide(dp_round, auditor, client, key_ipfs_address.multihash))

    @log(_logger, logging.INFO)
    def get_provided_decryption_keys(self, dp_round: int, auditor: Optional[EthereumAddress] = None, client: Optional[EthereumAddress] = None) -> Sequence[DecryptionKeyResponse]:
        from_block = self.get_data_retrieval_start_block(dp_round)
        to_block = self.get_score_decrypting_start_block(dp_round) - 1
        with self._pool.get() as (contract, dummy_wwwrapper):
            events = cast(Iterable[web3.types.EventData], contract.events.DecryptionKeyResponse.getLogs({"_client": client, "_auditor": auditor, "_dp_round": dp_round}, fromBlock=from_block, toBlock=to_block))
        answer: List[DecryptionKeyResponse] = []
        for event in events:
            answer.append(DecryptionKeyResponse(
                auditor=event['args']['auditor'],
                client=event['args']['client'],
                dp_round=dp_round,
                provider=event['args']['provider'],
                decryption_key_ipfs_address=event['args']['decryption_key_ipfs_address']
                ))
        return answer

    @log(_logger, logging.INFO)
    def enroll_client(self) -> HexBytes:
        bond_amount = self.bond_amount
        with self._pool.get() as (contract, wwwrapper):
            return wwwrapper.transact(contract.functions.enroll_client(), {'value': bond_amount})

    @log(_logger, logging.DEBUG, log_return=False)
    @lru_cache()
    def _get_client_enrollment_event(self, client: EthereumAddress) -> web3.types.EventData:
        from_block = max(0, self.start_block - self._submission_blockdelta)
        to_block = self.start_block - 1
        with self._pool.get() as (contract, dummy_wwwrapper):
            events = cast(Iterable[web3.types.EventData], contract.events.ClientEnrolled.getLogs({"_client": client}, fromBlock=from_block, toBlock=to_block))
        for event in events:
            return event
        raise EventNotFoundError(f"get_client_enrollment_event not found for client({client})")

    @log(_logger, logging.DEBUG, log_return=False)
    @lru_cache()
    def derive_ecdh_key(self, client: EthereumAddress) -> bytes:
        client_event = self._get_client_enrollment_event(client)
        transaction_digest: HexBytes = client_event['transactionHash']
        with self._pool.get() as (dummy_contract, wwwrapper):
            ecdh_key = wwwrapper.derive_ecdh_key(transaction_digest)
            assert isinstance(ecdh_key, bytes)
            return ecdh_key

    @log(_logger, logging.DEBUG)
    def get_client_dp_round(self, client: EthereumAddress) -> int:
        with self._pool.get() as (contract, dummy_wwwrapper):
            answer = contract.functions.get_client_dp_round(client).call()
            assert isinstance(answer, int)
            return answer

    @log(_logger, logging.DEBUG)
    def is_active_client(self, client: EthereumAddress) -> bool:
        with self._pool.get() as (contract, dummy_wwwrapper):
            answer = contract.functions.is_active_client(client).call()
            assert isinstance(answer, bool)
            return answer

    @log(_logger, logging.DEBUG)
    def is_client_training(self, client: EthereumAddress) -> bool:
        with self._pool.get() as (contract, dummy_wwwrapper):
            answer = contract.functions.is_client_training(client).call()
            assert isinstance(answer, bool)
            return answer

    @log(_logger, logging.DEBUG)
    def is_client_retrieving_data(self, client: EthereumAddress) -> bool:
        with self._pool.get() as (contract, dummy_wwwrapper):
            answer = contract.functions.is_client_retrieving_data(client).call()
            assert isinstance(answer, bool)
            return answer

    @log(_logger, logging.DEBUG)
    def is_client_scoring(self, client: EthereumAddress) -> bool:
        with self._pool.get() as (contract, dummy_wwwrapper):
            answer = contract.functions.is_client_scoring(client).call()
            assert isinstance(answer, bool)
            return answer

    @log(_logger, logging.DEBUG)
    def is_client_score_decrypting(self, client: EthereumAddress) -> bool:
        with self._pool.get() as (contract, dummy_wwwrapper):
            answer = contract.functions.is_client_score_decrypting(client).call()
            assert isinstance(answer, bool)
            return answer

    @lru_cache()
    def get_dp_round_start_block(self, dp_round: int) -> int:
        with self._pool.get() as (contract, dummy_wwwrapper):
            answer = contract.functions.get_dp_round_start_block(dp_round).call()
            assert isinstance(answer, int)
            return answer

    @lru_cache()
    def get_data_ipfs_addresses(self, dp_round: int) -> Mapping[EthereumAddress, cid.CIDv0]:
        assert self.get_latest_block_number() >= self.get_data_retrieval_start_block(dp_round), "please wait until training is finished to get the data ipfs addresses"
        from_block = self.get_training_submission_start_block(dp_round)
        to_block = self.get_data_retrieval_start_block(dp_round) - 1
        with self._pool.get() as (contract, dummy_wwwrapper):
            events = cast(Iterable[web3.types.EventData], contract.events.DataSubmitted.getLogs({"_dp_round": dp_round}, fromBlock=from_block, toBlock=to_block))
        answer: Dict[EthereumAddress, cid.CIDv0] = {}
        for event in events:
            answer[event['args']['client']] = cid.CIDv0(event['args']['data_ipfs_address'])
        return answer

    @log(_logger, logging.INFO)
    def submit_data(self, ipfs_address: cid.CIDv0) -> HexBytes:
        with self._pool.get() as (contract, wwwrapper):
            return wwwrapper.transact(contract.functions.submit_data(ipfs_address.buffer))

    @log(_logger, logging.INFO)
    @lru_cache()
    def get_submitted_data_event(self, dp_round: int, client: EthereumAddress) -> web3.types.EventData:
        from_block = self.get_training_submission_start_block(dp_round)
        to_block = self.get_data_retrieval_start_block(dp_round) - 1
        with self._pool.get() as (contract, dummy_wwwrapper):
            events = cast(Iterable[web3.types.EventData], contract.events.DataSubmitted.getLogs({"_dp_round": dp_round, "_client": client}, fromBlock=from_block, toBlock=to_block))
        for event in events:
            return event
        raise EventNotFoundError()

    @log(_logger, logging.INFO)
    def mark_data_retrieved(self, client: EthereumAddress) -> HexBytes:
        with self._pool.get() as (contract, wwwrapper):
            return wwwrapper.transact(contract.functions.mark_data_retrieved(client))

    @log(_logger, logging.INFO)
    @lru_cache()
    def get_mark_data_retrieved_event(self, dp_round: int, auditor: EthereumAddress, client: EthereumAddress) -> web3.types.EventData:
        from_block = self.get_data_retrieval_submission_start_block(dp_round)
        to_block = self.get_scoring_start_block(dp_round) - 1
        with self._pool.get() as (contract, dummy_wwwrapper):
            events = cast(Iterable[web3.types.EventData], contract.events.DataRetrieved.getLogs({"_client": client, "_auditor": auditor, "_dp_round": dp_round}, fromBlock=from_block, toBlock=to_block))
        for event in events:
            return event
        raise EventNotFoundError()

    @log(_logger, logging.INFO)
    def advance_to_scoring_stage(self) -> HexBytes:
        with self._pool.get() as (contract, wwwrapper):
            return wwwrapper.transact(contract.functions.advance_to_scoring_stage())

    @log(_logger, logging.INFO)
    @lru_cache()
    def get_data_retrieval_results(self, dp_round: int) -> Mapping[EthereumAddress, Tuple[int, int]]:
        assert self.get_latest_block_number() >= self.get_scoring_start_block(dp_round), "calling too soon"
        from_block = self.get_data_retrieval_submission_start_block(dp_round)
        to_block = self.get_scoring_start_block(dp_round) - 1
        with self._pool.get() as (contract, dummy_wwwrapper):
            events = cast(Iterable[web3.types.EventData], contract.events.DataRetrieved.getLogs({"_dp_round": dp_round}, fromBlock=from_block, toBlock=to_block))
        data_retrieval_results: Dict[EthereumAddress, Tuple[int, int]] = {}
        for event in events:
            auditor = event['args']['auditor']
            if auditor not in data_retrieval_results:
                data_retrieval_results[auditor] = 0, 0
            data_retrieval_results[auditor] = data_retrieval_results[auditor][0], data_retrieval_results[auditor][1] + 1
            client = event['args']['client']
            if client not in data_retrieval_results:
                data_retrieval_results[client] = 0, 0
            data_retrieval_results[client] = data_retrieval_results[client][0] + 1, data_retrieval_results[client][1]
        return data_retrieval_results

    @log(_logger, logging.INFO)
    def submit_encrypted_score(self, client: EthereumAddress, salt: bytes, score: float) -> HexBytes:
        score_int = int(score * self.score_denom)
        score_bytes = int_to_big_endian(score_int).rjust(32, b'\0')
        assert isinstance(score_bytes, bytes)
        assert len(score_bytes) == 32
        assert len(salt) == 32
        data_combined = salt + score_bytes
        encrypted_score = keccak(data_combined)
        with self._pool.get() as (contract, wwwrapper):
            if __debug__:
                chain_encrypted_score = contract.functions.encrypt_score(salt, score_int).call()
                assert encrypted_score == chain_encrypted_score, "incorrect score encryption"
            return wwwrapper.transact(contract.functions.submit_encrypted_score(client, encrypted_score))

    @log(_logger, logging.INFO)
    @lru_cache()
    def get_submitted_encrypted_score_event(self, dp_round: int, auditor: EthereumAddress, client: EthereumAddress) -> web3.types.EventData:
        from_block = self.get_scoring_submission_start_block(dp_round)
        to_block = self.get_score_decrypting_start_block(dp_round) - 1
        with self._pool.get() as (contract, dummy_wwwrapper):
            events = cast(Iterable[web3.types.EventData], contract.events.EncryptedScoreSubmitted.getLogs({"_dp_round": dp_round, "_auditor": auditor, "_client": client}, fromBlock=from_block, toBlock=to_block))
        for event in events:
            return event
        raise EventNotFoundError()

    @log(_logger, logging.INFO)
    def advance_to_score_decrypting_stage(self) -> HexBytes:
        with self._pool.get() as (contract, wwwrapper):
            return wwwrapper.transact(contract.functions.advance_to_score_decrypting_stage())

    @log(_logger, logging.INFO)
    def submit_decrypted_score(self, client: EthereumAddress, salt: bytes, score: float) -> HexBytes:
        score_int = int(score * self.score_denom)
        with self._pool.get() as (contract, wwwrapper):
            return wwwrapper.transact(contract.functions.submit_decrypted_score(client, salt, score_int))

    @log(_logger, logging.INFO)
    @lru_cache()
    def get_submitted_decrypted_score_event(self, dp_round: int, auditor: EthereumAddress, client: EthereumAddress) -> web3.types.EventData:
        from_block = self.get_score_decrypting_submission_start_block(dp_round)
        to_block = self.get_dp_round_start_block(dp_round + 1) - 1
        with self._pool.get() as (contract, dummy_wwwrapper):
            events = cast(Iterable[web3.types.EventData], contract.events.DecryptedScoreSubmitted.getLogs({"_dp_round": dp_round, "_auditor": auditor, "_client": client}, fromBlock=from_block, toBlock=to_block))
        for event in events:
            return event
        raise EventNotFoundError()

    @log(_logger, logging.INFO)
    @lru_cache()
    def get_scores(self, dp_round: int) -> Mapping[Tuple[EthereumAddress, EthereumAddress], Fraction]:
        from_block = self.get_score_decrypting_submission_start_block(dp_round)
        to_block = self.get_dp_round_start_block(dp_round + 1) - 1
        assert self.get_latest_block_number() > to_block, "please wait until all scores have been decrypted"
        with self._pool.get() as (contract, dummy_wwwrapper):
            events = cast(Iterable[web3.types.EventData], contract.events.DecryptedScoreSubmitted.getLogs({"_dp_round": dp_round}, fromBlock=from_block, toBlock=to_block))
        scores: Dict[Tuple[EthereumAddress, EthereumAddress], Fraction] = {}
        for event in events:
            client = event['args']['client']
            auditor = event['args']['auditor']
            score = Fraction(event['args']['decrypted_score'], self.score_denom)
            scores[auditor, client] = score
        return scores

    @log(_logger, logging.INFO, log_return=False)
    def tally_model_score(self, auditor: EthereumAddress, client: EthereumAddress, proposed_median: Fraction) -> HexBytes:
        assert self.score_denom % proposed_median.denominator == 0, "score denom is incorrect"
        with self._pool.get() as (contract, wwwrapper):
            return wwwrapper.transact(contract.functions.tally_model_score(auditor, client, proposed_median.numerator * self.score_denom // proposed_median.denominator))

    @log(_logger, logging.INFO)
    def get_tallied_auditors_for_client(self, dp_round: int, client: EthereumAddress, median_score: Fraction) -> Sequence[EthereumAddress]:
        assert self.score_denom % median_score.denominator == 0, "score denom is incorrect"
        from_block = self.get_dp_round_start_block(dp_round + 1)
        to_block = from_block + min(self._dp_round_training_blockdelta, self._submission_blockdelta) - 1
        addresses: List[EthereumAddress] = []
        with self._pool.get() as (contract, dummy_wwwrapper):
            events = cast(Iterable[web3.types.EventData], contract.events.ModelScoreTallied.getLogs({"_dp_round": dp_round, "_client": client, "_median_score": median_score.numerator * self.score_denom // median_score.denominator}, fromBlock=from_block, toBlock=to_block))
        for event in events:
            addresses.append(event['args']['auditor'])
        return addresses

    @log(_logger, logging.INFO)
    def commit_model_median_score(self, client: EthereumAddress, proposed_median: Fraction) -> HexBytes:
        assert self.score_denom % proposed_median.denominator == 0, "score denom is incorrect"
        with self._pool.get() as (contract, wwwrapper):
            return wwwrapper.transact(contract.functions.commit_model_median_score(client, proposed_median.numerator * self.score_denom // proposed_median.denominator))

    @log(_logger, logging.INFO)
    @lru_cache()
    def get_model_committed_median_score(self, dp_round: int, client: EthereumAddress) -> web3.types.EventData:
        from_block = self.get_dp_round_start_block(dp_round + 1)
        to_block = from_block + min(self._dp_round_training_blockdelta, self._submission_blockdelta) - 1
        with self._pool.get() as (contract, dummy_wwwrapper):
            events = cast(Iterable[web3.types.EventData], contract.events.ModelMedianScoreCommitted.getLogs({"_dp_round": dp_round, "_client": client}, fromBlock=from_block, toBlock=to_block))
        for event in events:
            return event
        raise EventNotFoundError()

    @log(_logger, logging.INFO)
    def tally_dataset_score(self, client: EthereumAddress) -> HexBytes:
        with self._pool.get() as (contract, wwwrapper):
            return wwwrapper.transact(contract.functions.tally_dataset_score(client))

    @log(_logger, logging.INFO)
    @lru_cache()
    def get_tallied_dataset_score_event(self, dp_round: int, auditor: EthereumAddress, client: EthereumAddress) -> web3.types.EventData:
        from_block = self.get_dp_round_start_block(dp_round + 1)
        to_block = from_block + min(self._dp_round_training_blockdelta, self._submission_blockdelta) - 1
        with self._pool.get() as (contract, dummy_wwwrapper):
            events = cast(Iterable[web3.types.EventData], contract.events.DatasetScoreTallied.getLogs({"_dp_round": dp_round, "_auditor": auditor, "_client": client}, fromBlock=from_block, toBlock=to_block))
        for event in events:
            return event
        raise EventNotFoundError()

    @log(_logger, logging.INFO)
    def advance_to_next_dp_round(self) -> HexBytes:
        with self._pool.get() as (contract, wwwrapper):
            return wwwrapper.transact(contract.functions.advance_to_next_dp_round())

    @log(_logger, logging.INFO)
    @lru_cache()
    def get_advanced_to_next_dp_round_event(self, dp_round: int, client: EthereumAddress) -> web3.types.EventData:
        from_block = self.get_dp_round_start_block(dp_round + 1)
        to_block = from_block + min(self._dp_round_training_blockdelta, self._submission_blockdelta) - 1
        with self._pool.get() as (contract, dummy_wwwrapper):
            events = cast(Iterable[web3.types.EventData], contract.events.AdvancedToNextDPRound.getLogs({"_dp_round": dp_round, "_client": client}, fromBlock=from_block, toBlock=to_block))
        for event in events:
            return event
        raise EventNotFoundError()

    @log(_logger, logging.WARNING)
    def boot_client(self, client: EthereumAddress) -> HexBytes:
        with self._pool.get() as (contract, wwwrapper):
            return wwwrapper.transact(contract.functions.boot_client(client))

    @log(_logger, logging.INFO)
    @lru_cache()
    def _get_booted_clients(self, dp_round: int) -> Set[EthereumAddress]:
        # Returns the clients booted on the specified round
        assert dp_round > 0, "booted clients not available until the 1st round (post enrollment)"
        # if you are asking for the current round, then we MUST be in the scoring stage.
        current_block = self.get_latest_block_number()
        current_round = self.get_dp_round(current_block)
        assert dp_round <= current_round, "cannot request in the future"
        if dp_round == current_round:
            passed_boot_period = current_block >= self.get_scoring_start_block(dp_round)
            assert passed_boot_period, "if getting booted clients for current round, you must be passed the boot period"
        from_block = self.get_data_retrieval_submission_start_block(dp_round)
        to_block = self.get_scoring_start_block(dp_round) - 1
        with self._pool.get() as (contract, dummy_wwwrapper):
            events = cast(Iterable[web3.types.EventData], contract.events.ClientBooted.getLogs({"_dp_round": dp_round}, fromBlock=from_block, toBlock=to_block))
        booted_clients: Set[EthereumAddress] = set()
        for event in events:
            booted_clients.add(event['args']['client'])
        return booted_clients

    @log(_logger, logging.INFO)
    @lru_cache()
    def get_active_clients(self, dp_round: int) -> Set[EthereumAddress]:
        assert dp_round > 0, "Client information not availabe at the beginning"
        if dp_round == 1:
            return set(self.enrolled_clients)  # for the first round, everyone's active
        current_block = self.get_latest_block_number()
        current_round = self.get_dp_round(current_block)
        assert dp_round <= current_round, "cannot request in the future"
        return self.get_active_clients(dp_round - 1) - self._get_booted_clients(dp_round - 1)
