import logging
import concurrent
import os
import tempfile
import tarfile
import secrets

import cid
from eth_typing.evm import BlockNumber

from ..client_config import ClientConfig
from ..model import ModelFactory
from ..ipfs_client import IPFSClient
from ..contract._contract_client import ContractClient
from .._utils.log_wrapper import log
from .._utils.encryption import encrypt_file, encrypt

class Train:
    _logger = logging.getLogger(__name__)

    def __init__(self, config: ClientConfig, model_factory: ModelFactory, contract_client: ContractClient, ipfs_client: IPFSClient, executor: concurrent.futures.Executor):
        self._config = config
        self._contract_client = contract_client
        self._executor = executor
        self._ipfs_client = ipfs_client
        self._model_factory = model_factory

    @log(_logger, logging.INFO, log_return=False)
    def _train(self, dp_round: int) -> None:
        submission_block = self._contract_client.get_data_retrieval_start_block(dp_round)
        train_deadline = self._contract_client.estimate_time_of_block(BlockNumber(submission_block - self._config.upload_buffer_blocks))
        self._model_factory.training_model.train(dp_round, train_deadline)

    @log(_logger, logging.INFO)
    def _submit_data(self, dp_round: int) -> cid.CIDv0:
        with tempfile.TemporaryDirectory() as tempdir:
            submission_tarfile_path = os.path.join(tempdir, "submission.tar.gz")
            with tarfile.open(submission_tarfile_path, "x:gz") as t_f:
                decrypted_model_filepath = os.path.join(tempdir, "model.dat")
                self._model_factory.training_model.save(decrypted_model_filepath)
                if self._config.enable_encryption:
                    with open(os.path.join(self._config.encryption_keys_folder, f"dp_round_{dp_round}.dat"), "xb") as f:
                        encryption_key = secrets.token_bytes(32)
                        f.write(encryption_key)
                        self._logger.debug("encryption key dp round %d: %s", dp_round, encryption_key)
                    encrypted_model_filepath = os.path.join(tempdir, "encrypted_model.dat")
                    encrypt_file(encryption_key, decrypted_model_filepath, encrypted_model_filepath)
                    t_f.add(encrypted_model_filepath, "encrypted_model.dat")
                    active_clients = self._contract_client.get_active_clients(dp_round)
                    for client in active_clients:
                        derived_key = self._contract_client.derive_ecdh_key(client)
                        client_encrypted_key_path = os.path.join(tempdir, f"{client}.dat")
                        with open(client_encrypted_key_path, "xb") as f:
                            f.write(encrypt(derived_key, encryption_key))
                        t_f.add(client_encrypted_key_path, f"{client}.dat")
                else:
                    t_f.add(decrypted_model_filepath, "decrypted_model.dat")
            ipfs_address = self._ipfs_client.upload(submission_tarfile_path)
        self._logger.info("Waiting for train submission start block")
        self._contract_client.wait_for_block(BlockNumber(self._contract_client.get_training_submission_start_block(dp_round)))
        txn_hash = self._contract_client.submit_data(ipfs_address)
        tx_receipt = self._contract_client.wait_for_tx_receipt(txn_hash)
        self._contract_client.validate_tx_receipt(tx_receipt)
        assert txn_hash == self._contract_client.get_submitted_data_event(dp_round, self._contract_client.address)['transactionHash']
        return ipfs_address

    @log(_logger, logging.INFO, log_return=False)
    def train(self, dp_round: int) -> None:
        training_end_block = self._contract_client.get_data_retrieval_start_block(dp_round)
        if self._contract_client.is_client_training(self._contract_client.address):
            self._train(dp_round)
            assert self._contract_client.get_latest_block_number() < training_end_block, "passed training deadline without submitting data"
            self._submit_data(dp_round)
            assert self._contract_client.is_client_retrieving_data(self._contract_client.address)
        self._logger.info("Waiting for training end block")
        self._contract_client.wait_for_block(training_end_block)
        assert self._contract_client.get_latest_block_number() >= training_end_block
