import os
import tempfile
import logging
import concurrent.futures
from typing import Optional, Type, Dict
from types import TracebackType
import shutil
import dataclasses

from eth_utils.hexadecimal import decode_hex
from eth_utils.currency import from_wei
import web3.types

from .model import ModelFactory, EvaluationModel
from .ipfs_client import IPFSClient
from .client_config import ClientConfig
from .contract.web3_factory import Web3Factory
from .contract.ethereum_address import EthereumAddress
from .contract.contract_factory import ContractFactory
from .statistics.base import BaseRecorder
from .contract._contract_client import ContractClient
from .contract._contract_wwwrapper_pool import ContractWWWrapperPool
from ._utils.log_wrapper import log
from ._stages.score import Score
from ._stages.score_decrypt import ScoreDecrypt
from ._stages.data_retrieval import DataRetrieval
from ._stages.train import Train
from ._stages.post_submission import PostSubmission
from .hooks import Hooks

class Experiment:
    _logger = logging.getLogger(__name__)

    def __init__(self,
                 model_factory: ModelFactory,
                 address: EthereumAddress,
                 config: ClientConfig,
                 web3_factory: Web3Factory,
                 ipfs_client: IPFSClient,
                 statistics_recorder: Optional[BaseRecorder] = None,
                 contract_factory: Optional[ContractFactory] = None,
                 hooks: Optional[Hooks] = None):
        self._ipfs_client = ipfs_client
        self._model_factory = model_factory
        self._web3_factory = web3_factory
        self._statistics_recorder = statistics_recorder
        if hooks is None:
            self._hooks = Hooks()
        else:
            self._hooks = hooks

        if contract_factory is None:
            self._contract_factory = ContractFactory()
        else:
            self._contract_factory = contract_factory

        if config.max_threads is None:
            # https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.ThreadPoolExecutor
            cpu_count = os.cpu_count()
            if cpu_count is None:
                cpu_count = 2
            max_threads = min(32, cpu_count + 4)
        else:
            max_threads = config.max_threads

        if config.max_web3_connections is None:
            max_web3_connections = max_threads
        else:
            max_web3_connections = config.max_web3_connections
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_threads)
        contract_wwwrapper_pool = ContractWWWrapperPool(max_web3_connections, decode_hex(config.private_key), address, self._web3_factory, self._contract_factory)

        self._contract_client = ContractClient(contract_wwwrapper_pool)
        self._train = Train(config, self._model_factory, self._contract_client, self._ipfs_client, self._executor)
        if config.retrieved_data_folder is None:
            self._retrieved_data_folder = tempfile.mkdtemp()
            self._retrieved_data_folder_is_temp = True
        else:
            self._retrieved_data_folder = config.retrieved_data_folder
            self._retrieved_data_folder_is_temp = False
        self._data_retrieval = DataRetrieval(self._ipfs_client, self._model_factory, self._contract_client, self._executor, self._retrieved_data_folder)
        self._score = Score(self._model_factory, config.encryption_keys_folder, self._contract_client, self._executor, self._statistics_recorder)
        self._score_decrypt = ScoreDecrypt(config, self._contract_client, self._executor)
        self._post_submission = PostSubmission(config, self._contract_client, self._model_factory, self._executor, statistics_recorder)

    def __enter__(self) -> 'Experiment':
        return self

    def close(self) -> None:
        self._executor.shutdown(wait=True)
        if self._statistics_recorder is not None:
            self._statistics_recorder.close()
        if self._retrieved_data_folder_is_temp:
            shutil.rmtree(self._retrieved_data_folder, ignore_errors=True)

    def __exit__(self, exc_type: Optional[Type[BaseException]], exc_val: Optional[BaseException], exc_tb: Optional[TracebackType]) -> None:
        self.close()

    @log(_logger, logging.INFO)
    def _enroll_client(self) -> None:
        txn_hash = self._contract_client.enroll_client()
        tx_receipt = self._contract_client.wait_for_tx_receipt(txn_hash)
        self._contract_client.validate_tx_receipt(tx_receipt)
        assert self._contract_client.is_active_client(self._contract_client.address)

    @log(_logger, logging.INFO)
    def _record_gas(self, dp_round: int, previous_nonce: web3.types.Nonce) -> web3.types.Nonce:
        new_nonce = self._contract_client.get_current_nonce()
        if self._statistics_recorder is not None:
            gas = 0
            for nonce in range(previous_nonce, new_nonce):
                gas += self._contract_client.wait_for_tx_receipt(self._contract_client.get_txn_hash(web3.types.Nonce(nonce)))['gasUsed']
            self._statistics_recorder.record_scalar(dp_round, "gas", gas)
        return new_nonce

    @log(_logger, logging.INFO, log_return=False)
    def run(self) -> None:
        previous_nonce = self._contract_client.get_current_nonce()
        round_0_gwei = 0
        if not self._contract_client.is_active_client(self._contract_client.address):
            # wait for enrollment
            self._logger.info("Waiting for enrollment start block: %d", self._contract_client.enrollment_start_block)
            self._contract_client.wait_for_block(self._contract_client.enrollment_start_block)
            round_0_gwei = int(from_wei(self._contract_client.get_balance(), 'gwei'))
            if self._statistics_recorder is not None:
                self._statistics_recorder.record_scalar(0, "starting_balance_gwei", 0)  # recording the diff
            self._hooks.pre_enrollment(self)
            self._enroll_client()
            self._hooks.post_enrollment(self)
            if self._statistics_recorder is not None:
                round_1_gwei = int(from_wei(self._contract_client.get_balance(), 'gwei'))
                self._statistics_recorder.record_scalar(1, "starting_balance_gwei", round_1_gwei - round_0_gwei)
            previous_nonce = self._record_gas(dp_round=0, previous_nonce=previous_nonce)
        start_block = self._contract_client.start_block
        latest_block_number = self._contract_client.get_latest_block_number()
        if latest_block_number < start_block:
            self._logger.info("Waiting for start block")
            self._contract_client.wait_for_block(start_block)
            latest_block_number = self._contract_client.get_latest_block_number()
            assert latest_block_number >= start_block
        dp_round = self._contract_client.get_dp_round(latest_block_number)
        assert dp_round > 0, "dp round should be greater than 0 since we're passed enrollment"
        if self._statistics_recorder is not None:
            for param_name, param_value in dataclasses.asdict(self._contract_client.parameters).items():
                self._statistics_recorder.record_text(0, param_name, str(param_value))
        data: Optional[Dict[EthereumAddress, EvaluationModel]] = None
        while True:
            my_dp_round = self._contract_client.get_client_dp_round(self._contract_client.address)
            if my_dp_round < dp_round:
                assert my_dp_round == dp_round - 1, "you have fallen too far behind and will be booted"
                self._hooks.pre_post_submit(my_dp_round, self)
                if data is None:
                    data = self._data_retrieval.data_retrieval(my_dp_round)
                self._post_submission.post_submit(my_dp_round, data)
                self._hooks.post_post_submit(my_dp_round, self)
                previous_nonce = self._record_gas(my_dp_round, previous_nonce)
                my_dp_round = self._contract_client.get_client_dp_round(self._contract_client.address)
            assert my_dp_round == dp_round, "failed to advance self to the current state"
            if dp_round > self._contract_client.num_dp_rounds:
                # breaking here so we will always attempt to submit consensus scores, even if we crash on the last round and restart
                self._logger.info("Waiting for pseudo data retrieval start block so we can collect the final reward")
                self._contract_client.wait_for_block(self._contract_client.get_data_retrieval_start_block(dp_round))
                self._hooks.pre_collect_reward(dp_round, self)
                txn_hash = self._contract_client.collect_reward(dp_round - 1)
                txn_receipt = self._contract_client.wait_for_tx_receipt(txn_hash)
                self._contract_client.validate_tx_receipt(txn_receipt)
                self._hooks.post_collect_reward(dp_round, self)
                previous_nonce = self._record_gas(dp_round, previous_nonce)  # record this as the gas for the pseudo round
                if self._statistics_recorder is not None:
                    self._statistics_recorder.record_scalar(dp_round, "starting_balance_gwei", int(from_wei(self._contract_client.get_balance(), 'gwei')) - round_0_gwei)
                break
            if self._statistics_recorder is not None:
                self._statistics_recorder.record_scalar(dp_round, "num_active_clients", len(self._contract_client.get_active_clients(dp_round)))
            self._hooks.pre_train(dp_round, self)
            self._train.train(dp_round)
            self._hooks.post_train(dp_round, self)
            if dp_round > 1:
                self._hooks.pre_collect_reward(dp_round, self)
                txn_hash = self._contract_client.collect_reward(dp_round - 1)
                txn_receipt = self._contract_client.wait_for_tx_receipt(txn_hash)
                self._contract_client.validate_tx_receipt(txn_receipt)
                self._hooks.post_collect_reward(dp_round, self)
                if self._statistics_recorder is not None:
                    self._statistics_recorder.record_scalar(dp_round, "starting_balance_gwei", int(from_wei(self._contract_client.get_balance(), 'gwei')) - round_0_gwei)
            self._hooks.pre_data_retrieval(dp_round, self)
            data = self._data_retrieval.data_retrieval(dp_round)
            self._hooks.post_data_retrieval(dp_round, self, data)
            self._hooks.pre_score(dp_round, self)
            scores = self._score.score(dp_round, data)
            self._hooks.post_score(dp_round, self, scores)
            self._hooks.pre_score_decrypt(dp_round, self)
            self._score_decrypt.score_decrypt(dp_round, scores)
            self._hooks.post_score_decrypt(dp_round, self)
            dp_round += 1
