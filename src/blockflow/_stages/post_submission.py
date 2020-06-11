from typing import Dict, List, Optional, Sequence, Mapping
import logging
import concurrent
from fractions import Fraction
import time

import numpy as np
from hexbytes.main import HexBytes

from ..model import EvaluationModel, ModelFactory, OthersWork
from ..contract.ethereum_address import EthereumAddress
from ..client_config import ClientConfig
from ..contract._contract_client import ContractClient, EventNotFoundError
from .._utils.log_wrapper import log
from ..statistics.base import BaseRecorder
from .._utils.futures import wait_for_all

class PostSubmission:
    _logger = logging.getLogger(__name__)

    def __init__(self, config: ClientConfig, contract_client: ContractClient, model_factory: ModelFactory, executor: concurrent.futures.Executor, statistics_recorder: Optional[BaseRecorder] = None):
        self._contract_client = contract_client
        self._config = config
        self._executor = executor
        self._statistics_recorder = statistics_recorder
        self._model_factory = model_factory

    @log(_logger, logging.INFO)
    def _did_advance_client_to_commit(self, dp_round: int, auditors: Sequence[EthereumAddress], client: EthereumAddress, median_score: Fraction) -> bool:
        submitted_auditors = self._contract_client.get_tallied_auditors_for_client(dp_round, client, median_score)
        remaining_auditors = set(auditors) - set(submitted_auditors)
        if len(remaining_auditors) == 0:
            try:
                if not self._config.tally_self_only or self._contract_client.address == client:
                    self._contract_client.commit_model_median_score(client, median_score)
            except:
                # there might be an error with a race condition, where multiple clients realize that it can be committed. in this case, we should require that:
                if self._is_median_score_committed(dp_round, client):
                    return True
                raise
            else:
                return True
        self._logger.debug("For client(%s), median_score(%s), still waiting for the following auditors %s", client, median_score, remaining_auditors)
        return False

    @log(_logger, logging.INFO)
    def _is_median_score_committed(self, dp_round: int, client: EthereumAddress) -> bool:
        try:
            self._contract_client.get_model_committed_median_score(dp_round, client)
        except EventNotFoundError:
            return False
        else:
            return True

    @log(_logger, logging.INFO)
    def _tally_dataset_score(self, dp_round: int, client: EthereumAddress) -> HexBytes:
        try:
            return self._contract_client.tally_dataset_score(client)
        except Exception as ex:
            try:
                event = self._contract_client.get_tallied_dataset_score_event(dp_round, self._contract_client.address, client)
                txn_hash = event['transactionHash']
                assert isinstance(txn_hash, HexBytes)
                return txn_hash
            except EventNotFoundError:
                raise ex

    @log(_logger, logging.INFO, log_return=False)
    def _wait_for_dataset_score_tallied(self, dp_round: int, txn_hash: HexBytes, client: EthereumAddress) -> None:
        tx_receipt = self._contract_client.wait_for_tx_receipt(txn_hash)
        self._contract_client.validate_tx_receipt(tx_receipt)
        if __debug__ or self._statistics_recorder is not None:
            event = self._contract_client.get_tallied_dataset_score_event(dp_round, self._contract_client.address, client)
            assert txn_hash == event['transactionHash']
            if self._statistics_recorder is not None:
                score = Fraction(event['args']['dataset_score'], self._contract_client.score_denom)
                self._statistics_recorder.record_scalars(dp_round, "dataset_score", {client: float(score)})

    @log(_logger, logging.INFO, log_return=False)
    def _update_model_factory(self, dp_round: int, scores: Mapping[EthereumAddress, float], data: Mapping[EthereumAddress, EvaluationModel]) -> None:
        others_work: List[OthersWork] = []
        assert len(scores) == len(data), "invariant violation"
        for client, score in scores.items():
            model = data[client]
            others_work.append(OthersWork(client, model, score))
        self._model_factory.training_model.update(dp_round, others_work)

    @log(_logger, logging.INFO, log_return=False)
    def post_submit(self, dp_round: int, models: Mapping[EthereumAddress, EvaluationModel]) -> None:
        scores = self._contract_client.get_scores(dp_round)
        client_to_scores: Dict[EthereumAddress, List[int]] = {}
        client_to_auditors: Dict[EthereumAddress, List[EthereumAddress]] = {}
        client_to_median: Dict[EthereumAddress, Fraction] = {}
        client_to_median_float: Dict[EthereumAddress, float] = {}
        for (auditor, client), score in scores.items():
            if client not in client_to_scores:
                client_to_scores[client] = []
            if client not in client_to_auditors:
                client_to_auditors[client] = []
            assert self._contract_client.score_denom % score.denominator == 0
            client_to_scores[client].append(score.numerator * self._contract_client.score_denom / score.denominator)
            client_to_auditors[client].append(auditor)
        for client, client_scores in client_to_scores.items():
            median_score = Fraction(int(np.percentile(np.array(client_scores, dtype=np.int), q=50, interpolation='lower')), self._contract_client.score_denom)
            client_to_median[client] = median_score
            client_to_median_float[client] = float(median_score)

        self._update_model_factory(dp_round, client_to_median_float, models)

        if self._statistics_recorder is not None:
            self._statistics_recorder.record_scalars(dp_round, "median_scores", client_to_median_float)
            received_scores: Dict[EthereumAddress, float] = {}
            for client, received_score in zip(client_to_auditors[self._contract_client.address], client_to_scores[self._contract_client.address]):
                received_scores[client] = received_score / self._contract_client.score_denom
            self._statistics_recorder.record_scalars(dp_round, "received_scores", received_scores)

        # fire and forgetting since most of the time this will be a race condition, and the transaction will fail
        futs: List[concurrent.futures.Future[HexBytes]] = []
        for (auditor, client) in scores.keys():
            if (not self._config.tally_self_only) or auditor == self._contract_client.address:
                fut = self._executor.submit(self._contract_client.tally_model_score, auditor, client, client_to_median[client])
                futs.append(fut)
        if self._config.tally_self_only:
            # if we're only submitting for me myself and I, then all of this should go through. Let's wait to capture any errors
            tally_txn_hash = wait_for_all(futs)
            for txn_hash in tally_txn_hash:
                tx_receipt = self._contract_client.wait_for_tx_receipt(txn_hash)
                self._contract_client.validate_tx_receipt(tx_receipt)

        remaining_clients_to_advance = list(client_to_scores.keys())
        remaining_clients_to_commit: List[EthereumAddress] = []
        while len(remaining_clients_to_advance) + len(remaining_clients_to_commit) > 0:
            if self._contract_client.get_latest_block_number() >= self._contract_client.get_data_retrieval_start_block(dp_round + 1):
                raise Exception("passed the data retrieval start block without finishing the tally")
            did_advance_client_futs: List[concurrent.futures.Future[bool]] = []
            for client in remaining_clients_to_advance:
                did_advance_client_futs.append(self._executor.submit(self._did_advance_client_to_commit, dp_round, client_to_auditors[client], client, client_to_median[client]))

            did_advance_clients = wait_for_all(did_advance_client_futs)
            remaining_clients_to_advance_new = []
            for client, did_advance in zip(remaining_clients_to_advance, did_advance_clients):
                if did_advance:
                    remaining_clients_to_commit.append(client)
                else:
                    remaining_clients_to_advance_new.append(client)
            remaining_clients_to_advance = remaining_clients_to_advance_new

            is_median_score_committed_futs: List[concurrent.futures.Future[bool]] = []
            for client in remaining_clients_to_commit:
                is_median_score_committed_futs.append(self._executor.submit(self._is_median_score_committed, dp_round, client))

            is_median_score_committed = wait_for_all(is_median_score_committed_futs)
            remaining_clients_to_commit_new = []
            for client, did_commit in zip(remaining_clients_to_commit, is_median_score_committed):
                if not did_commit:
                    remaining_clients_to_commit_new.append(client)
            remaining_clients_to_commit = remaining_clients_to_commit_new
            time.sleep(1.0)
        self._logger.info("Finished tallying clients")
        # once we have everthing committed, then we can tally our dataset scores
        tally_dataset_score_transaction_futs: List[concurrent.futures.Future[HexBytes]] = []
        clients_ordered = list(client_to_median.keys())
        for client in clients_ordered:
            tally_dataset_score_transaction_futs.append(self._executor.submit(self._tally_dataset_score, dp_round, client))
        txn_hshes = wait_for_all(tally_dataset_score_transaction_futs)

        # wait for the tallies to be mined
        tally_dataset_mined_futs: List[concurrent.futures.Future[None]] = []
        for txn_hsh, client in zip(txn_hshes, clients_ordered):
            tally_dataset_mined_futs.append(self._executor.submit(self._wait_for_dataset_score_tallied, dp_round, txn_hsh, client))
        wait_for_all(tally_dataset_mined_futs)

        # then advance to the future
        txn_hash = self._contract_client.advance_to_next_dp_round()
        tx_receipt = self._contract_client.wait_for_tx_receipt(txn_hash)
        self._contract_client.validate_tx_receipt(tx_receipt)
        # advancement_event = self._contract_client.get_advanced_to_next_dp_round_event(dp_round, self._contract_client.address)
        # my_score = Fraction(advancement_event['args']['score'], self._contract_client.score_denom)
        # self._logger.info("My overall score for dp round %d is %s", dp_round, my_score)
        # if self._statistics_recorder is not None:
        #     self._statistics_recorder.record_scalar(dp_round, "overall_score", float(my_score))
