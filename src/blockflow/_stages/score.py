from typing import Mapping, Sequence, Dict, List, Optional
import math
import logging
import concurrent
import secrets
import os

import cid
from hexbytes.main import HexBytes
from eth_typing.evm import BlockNumber

from ..model import EvaluationModel, ModelFactory
from ..contract.ethereum_address import EthereumAddress
from ..contract._contract_client import ContractClient, EventNotFoundError
from .._utils.log_wrapper import log
from ..statistics.base import BaseRecorder
from .._utils.futures import wait_for_all

class Score:
    _logger = logging.getLogger(__name__)

    def __init__(self, model_factory: ModelFactory, encryption_keys_folder: str, contract_client: ContractClient, executor: concurrent.futures.Executor, statistics_recorder: Optional[BaseRecorder] = None):
        self._encryption_keys_folder = encryption_keys_folder
        self._contract_client = contract_client
        self._executor = executor
        self._model_factory = model_factory
        self._statistics_recorder = statistics_recorder

    @log(_logger, logging.INFO)
    def _compute_scores(self, dp_round: int, data: Mapping[EthereumAddress, EvaluationModel]) -> Mapping[EthereumAddress, float]:
        scores: Dict[EthereumAddress, float] = {}
        for client, model in data.items():
            self._logger.debug("Scoring model for client %s", client)
            # Not parallelizing since model.score is already multithreaded
            scores[client] = model.score()
        if self._statistics_recorder is not None:
            self._statistics_recorder.record_scalars(dp_round, "my_dataset_their_model", scores)
        return scores

    @log(_logger, logging.INFO)
    def _get_clients_to_include(self, dp_round: int, clients_submitted_data: Sequence[EthereumAddress]) -> Sequence[EthereumAddress]:
        data_retrieval_results = self._contract_client.get_data_retrieval_results(dp_round)
        cutoff = math.floor(len(clients_submitted_data) * self._contract_client.min_agreement_threshold)
        self._logger.debug("DP round %d has inclusion cutoff threshold %d", dp_round, cutoff)
        clients_to_include: List[EthereumAddress] = []
        for client, (auditor_score, client_score) in data_retrieval_results.items():
            if auditor_score >= cutoff and client_score >= cutoff:
                clients_to_include.append(client)
        return clients_to_include

    @log(_logger, logging.INFO, log_return=False)
    def _submit_client_encrypted_score(self, dp_round: int, client: EthereumAddress, score: float) -> HexBytes:
        try:
            event = self._contract_client.get_submitted_encrypted_score_event(dp_round, self._contract_client.address, client)
            txn_hash = event['transactionHash']
            assert isinstance(txn_hash, HexBytes)
            return txn_hash
        except EventNotFoundError:
            salt = secrets.token_bytes(32)
            salt_filename = os.path.join(self._encryption_keys_folder, f"salt_dp_round_{dp_round}_client_{client}.dat")
            with open(salt_filename, 'xb') as f:
                f.write(salt)
            return self._contract_client.submit_encrypted_score(client, salt, score)

    @log(_logger, logging.INFO, log_return=False)
    def _wait_for_client_encrypted_score(self, dp_round: int, txn_hash: HexBytes, client: EthereumAddress) -> None:
        tx_receipt = self._contract_client.wait_for_tx_receipt(txn_hash)
        self._contract_client.validate_tx_receipt(tx_receipt)
        if __debug__:
            retrieved_txn_hash = self._contract_client.get_submitted_encrypted_score_event(dp_round, self._contract_client.address, client)['transactionHash']
            assert txn_hash == retrieved_txn_hash

    @log(_logger, logging.INFO, log_return=False)
    def _submit_encrypted_scores(self, dp_round: int, scores: Mapping[EthereumAddress, float]) -> None:
        futures: List[concurrent.futures.Future[HexBytes]] = []
        for client, score in scores.items():
            futures.append(self._executor.submit(self._submit_client_encrypted_score, dp_round, client, score))
        txn_hshes = wait_for_all(futures)
        futures_2: List[concurrent.futures.Future[None]] = []
        for client, txn_hash in zip(scores.keys(), txn_hshes):
            futures_2.append(self._executor.submit(self._wait_for_client_encrypted_score, dp_round, txn_hash, client))
        wait_for_all(futures_2)

    @log(_logger, logging.INFO, log_return=False)
    def score(self, dp_round: int, data: Dict[EthereumAddress, EvaluationModel]) -> Mapping[EthereumAddress, float]:
        client_to_ipfs_folders = self._contract_client.get_data_ipfs_addresses(dp_round)
        scoring_end_block = self._contract_client.get_score_decrypting_start_block(dp_round)
        if self._contract_client.is_client_retrieving_data(self._contract_client.address):
            current_block = self._contract_client.get_latest_block_number()
            self._logger.debug("Block delta of %d to advance client to scoring", scoring_end_block - current_block)
            txn_hash = self._contract_client.advance_to_scoring_stage()
            tx_receipt = self._contract_client.wait_for_tx_receipt(txn_hash)
            self._contract_client.validate_tx_receipt(tx_receipt)
            assert self._contract_client.is_client_scoring(self._contract_client.address)
        # First remove clients who did not submit data to figure out who to include in the score
        clients_to_include = self._get_clients_to_include(dp_round, tuple(client_to_ipfs_folders.keys()))
        included_clients_missing_data_to_ipfs_folder: Dict[EthereumAddress, cid.CIDv0] = {}
        for client in clients_to_include:
            if client not in data:
                included_clients_missing_data_to_ipfs_folder[client] = client_to_ipfs_folders[client]
        assert len(included_clients_missing_data_to_ipfs_folder) == 0, "still missing data!!! Cannot score properly with missing data"
        final_data: Dict[EthereumAddress, EvaluationModel] = {}
        for client in clients_to_include:
            final_data[client] = data[client]
        self._logger.info("Filtered out clients that should not be included (i.e. failed to have concensus decryption)")
        scores = self._compute_scores(dp_round, final_data)
        self._logger.info("Waiting for scoring submission start block")
        self._contract_client.wait_for_block(BlockNumber(self._contract_client.get_scoring_submission_start_block(dp_round)))
        if self._contract_client.is_client_scoring(self._contract_client.address) and self._contract_client.get_latest_block_number() < scoring_end_block:
            self._submit_encrypted_scores(dp_round, scores)
        self._logger.info("Waiting for scoring end block")
        self._contract_client.wait_for_block(scoring_end_block)
        assert self._contract_client.get_latest_block_number() >= scoring_end_block
        return scores
