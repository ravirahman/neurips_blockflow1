from typing import Mapping, List
import logging
import concurrent
import os

from hexbytes.main import HexBytes
from eth_typing.evm import BlockNumber

from ..client_config import ClientConfig
from ..contract.ethereum_address import EthereumAddress
from ..contract._contract_client import ContractClient, EventNotFoundError
from .._utils.log_wrapper import log
from .._utils.futures import wait_for_all

class ScoreDecrypt:
    _logger = logging.getLogger(__name__)

    def __init__(self, config: ClientConfig, contract_client: ContractClient, executor: concurrent.futures.Executor):
        self._config = config
        self._contract_client = contract_client
        self._executor = executor

    @log(_logger, logging.INFO, log_return=False)
    def _decrypt_client_score(self, dp_round: int, client: EthereumAddress, score: float) -> HexBytes:
        try:
            txn_hash = self._contract_client.get_submitted_decrypted_score_event(dp_round, self._contract_client.address, client)['transactionHash']
            assert isinstance(txn_hash, HexBytes)
            return txn_hash
        except EventNotFoundError:
            salt_filename = os.path.join(self._config.encryption_keys_folder, f"salt_dp_round_{dp_round}_client_{client}.dat")
            with open(salt_filename, 'rb') as f:
                salt = f.read()
            return self._contract_client.submit_decrypted_score(client, salt, score)

    @log(_logger, logging.INFO, log_return=False)
    def _wait_for_decrypted_score(self, dp_round: int, txn_hash: HexBytes, client: EthereumAddress) -> None:
        tx_receipt = self._contract_client.wait_for_tx_receipt(txn_hash)
        self._contract_client.validate_tx_receipt(tx_receipt)
        if __debug__:
            event_txn_hash = self._contract_client.get_submitted_decrypted_score_event(dp_round, self._contract_client.address, client)['transactionHash']
            assert event_txn_hash == txn_hash

    @log(_logger, logging.INFO, log_return=False)
    def _decrypt_scores(self, dp_round: int, scores: Mapping[EthereumAddress, float]) -> None:
        futures: List[concurrent.futures.Future[HexBytes]] = []
        for client, score in scores.items():
            futures.append(self._executor.submit(self._decrypt_client_score, dp_round, client, score))
        txn_hshes = wait_for_all(futures)
        futures_2: List[concurrent.futures.Future[None]] = []
        for client, txn_hash in zip(scores.keys(), txn_hshes):
            futures_2.append(self._executor.submit(self._wait_for_decrypted_score, dp_round, txn_hash, client))
        wait_for_all(futures_2)

    @log(_logger, logging.INFO, log_return=False)
    def score_decrypt(self, dp_round: int, scores: Mapping[EthereumAddress, float]) -> None:
        score_decrypting_end_block = BlockNumber(self._contract_client.get_dp_round_start_block(dp_round + 1))
        if self._contract_client.is_client_scoring(self._contract_client.address):
            txn_hash = self._contract_client.advance_to_score_decrypting_stage()
            tx_receipt = self._contract_client.wait_for_tx_receipt(txn_hash)
            self._contract_client.validate_tx_receipt(tx_receipt)
            assert self._contract_client.is_client_score_decrypting(self._contract_client.address)
        self._logger.info("Waiting for score decrypting submission start block")
        self._contract_client.wait_for_block(BlockNumber(self._contract_client.get_score_decrypting_submission_start_block(dp_round)))
        if self._contract_client.is_client_score_decrypting(self._contract_client.address) and self._contract_client.get_latest_block_number() < score_decrypting_end_block:
            self._decrypt_scores(dp_round, scores)
        self._logger.info("Waiting for score decrypting end block")
        self._contract_client.wait_for_block(score_decrypting_end_block)
        assert self._contract_client.get_latest_block_number() >= score_decrypting_end_block
        # Should be at the next dp round
        assert self._contract_client.get_dp_round(self._contract_client.get_latest_block_number()) == dp_round + 1
