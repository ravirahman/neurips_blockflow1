from typing import Mapping, List, Dict, Optional
import logging
import concurrent
import os
import time
import tarfile

from hexbytes.main import HexBytes
import cid
from eth_typing.evm import BlockNumber

from ..model import EvaluationModel, ModelFactory
from ..ipfs_client import IPFSClient
from ..contract.ethereum_address import EthereumAddress
from ..contract._contract_client import ContractClient, EventNotFoundError, RevertException
from .._utils.log_wrapper import log
from .._utils.encryption import decrypt, decrypt_file
from .._utils.futures import wait_for_all

class DataRetrieval:
    _logger = logging.getLogger(__name__)

    def __init__(self, ipfs_client: IPFSClient, model_factory: ModelFactory, contract_client: ContractClient, executor: concurrent.futures.Executor, retrieved_data_folder: str):
        self._ipfs_client = ipfs_client
        self._contract_client = contract_client
        self._executor = executor
        self._model_factory = model_factory
        self._retrieved_data_folder = retrieved_data_folder

    @log(_logger, logging.INFO, log_return=False)
    def _boot_client_if_inactive(self, dp_round: int, client: EthereumAddress) -> None:
        client_dp_round = self._contract_client.get_client_dp_round(client)
        if client_dp_round < dp_round:
            try:
                txn_hash = self._contract_client.boot_client(client)
                tx_receipt = self._contract_client.wait_for_tx_receipt(txn_hash)
                self._contract_client.validate_tx_receipt(tx_receipt)
            except (ValueError, RevertException):
                if __debug__:
                    while True:
                        if not self._contract_client.is_active_client(client):
                            self._logger.debug("Client %s is no longer active", client)
                            return
                        self._logger.debug("Failed to boot client %s likely due to a race condition; going to sleep 1.0s and check to see if they are still active", client)
                        time.sleep(1.0)
                self._logger.warning("Failed to boot client %s, likely because they were already booted", client)
            else:
                self._logger.info("Booted client %s", client)
                assert not self._contract_client.is_active_client(client), "client not booted"
        else:
            self._logger.debug("Client %s is active", client)

    @log(_logger, logging.INFO, log_return=False)
    def _boot_inactive_clients(self, dp_round: int) -> None:
        active_clients = list(self._contract_client.get_active_clients(dp_round))
        futures: List[concurrent.futures.Future[None]] = []
        for client in active_clients:
            futures.append(self._executor.submit(self._boot_client_if_inactive, dp_round, client))
        for future in futures:
            future.result()

    @log(_logger, logging.INFO, log_return=False)
    def _retrieve_client_data(self, dp_round: int, client: EthereumAddress, dp_round_ipfs_submission_cid: cid.CIDv0, data: Dict[EthereumAddress, EvaluationModel],
                              shared_key_provider: Optional[EthereumAddress] = None, shared_key_ipfs_cid: Optional[cid.CIDv0] = None) -> None:
        dp_round_folder = os.path.join(self._retrieved_data_folder, f"client_{client}", f"dp_round_{dp_round}")
        os.makedirs(dp_round_folder)
        encrypted_submission_tarball_path = os.path.join(dp_round_folder, "encrypted_submission.tar.gz")
        encrypted_model_path = os.path.join(dp_round_folder, "encrypted_model.dat")
        decrypted_model_path = os.path.join(dp_round_folder, "decrypted_model.dat")
        shared_key_path = os.path.join(dp_round_folder, "shared_key.dat")

        encrypted_key: Optional[bytes] = None
        if shared_key_provider is None:
            assert shared_key_ipfs_cid is None, "shared_key_ipfs_cid must be None if shared_key_provider is None"
            shared_key_provider = client
        else:
            assert shared_key_ipfs_cid is not None, "shared_key_ipfs_cid must not be None if shared_key_ipfs_cid is not None"
            encrypted_key = self._ipfs_client.download(shared_key_path)

        if not os.path.exists(encrypted_model_path) or encrypted_key is None:
            if not os.path.exists(encrypted_submission_tarball_path):
                if not isinstance(dp_round_ipfs_submission_cid, cid.CIDv0):
                    raise Exception(f"dp_round_ipfs_submission_cid cid is invalid: {dp_round_ipfs_submission_cid}")
                # TODO validate size before download
                self._ipfs_client.download_to_file(f"/ipfs/{dp_round_ipfs_submission_cid}", encrypted_submission_tarball_path)
            with tarfile.open(encrypted_submission_tarball_path, "r:gz") as t_f:
                if not os.path.exists(encrypted_model_path):
                    try:
                        model_file_info = t_f.getmember("decrypted_model.dat")
                        model_path = decrypted_model_path
                    except KeyError:
                        model_file_info = t_f.getmember("encrypted_model.dat")
                        model_path = encrypted_model_path
                    if not model_file_info.isfile():
                        raise Exception(f"the model inside the tarfile is not a file")
                    t_f._extract_member(model_file_info, model_path, set_attrs=False)  # type: ignore # pylint: disable=protected-access

                if encrypted_key is None and not os.path.exists(decrypted_model_path):
                    shared_key_file_info = t_f.getmember(f"{self._contract_client.address}.dat")
                    if not shared_key_file_info.isfile():
                        raise Exception(f"shared_key_file_info {self._contract_client.address}.dat inside the tarfile is not a file")
                    encrypted_key_buffer = t_f.extractfile(shared_key_file_info)
                    assert encrypted_key_buffer is not None
                    encrypted_key = encrypted_key_buffer.read()
            if not os.path.exists(decrypted_model_path):
                derived_key = self._contract_client.derive_ecdh_key(shared_key_provider)
                assert encrypted_key is not None
                shared_key = decrypt(derived_key, encrypted_key)
                with open(shared_key_path, "wb+") as fp:  # overwrite in case we were wrong the first time
                    fp.write(shared_key)
                decrypt_file(shared_key, encrypted_model_path, decrypted_model_path)
            model = self._model_factory.load(decrypted_model_path)
            data[client] = model

    @log(_logger, logging.INFO, log_return=False)
    def _retrieve_data(self, dp_round: int, client_to_ipfs_folders: Mapping[EthereumAddress, cid.CIDv0], models: Dict[EthereumAddress, EvaluationModel]) -> None:
        futures: List[concurrent.futures.Future[None]] = []
        for client, ipfs_folder in client_to_ipfs_folders.items():
            futures.append(self._executor.submit(self._retrieve_client_data, dp_round, client, ipfs_folder, models))
        # Safe to ignore exceptions. Ideally limit to client errors. It is expected for the called method to throw if there is an issue with another client's submission
        concurrent.futures.wait(futures)

    @log(_logger, logging.INFO, log_return=False)
    def _mark_client_as_retrieved(self, dp_round: int, client: EthereumAddress) -> HexBytes:
        try:
            return self._contract_client.mark_data_retrieved(client)
        except Exception as ex:
            try:
                event = self._contract_client.get_mark_data_retrieved_event(dp_round, self._contract_client.address, client)
                self._logger.info("Client %s already marked as retrieved", client)
                txn_hash = event['transactionHash']
                assert isinstance(txn_hash, HexBytes)
                return txn_hash
            except EventNotFoundError:
                raise ex

    @log(_logger, logging.INFO, log_return=False)
    def _wait_for_client_retrieved(self, dp_round: int, txn_hash: HexBytes, client: EthereumAddress) -> None:
        tx_receipt = self._contract_client.wait_for_tx_receipt(txn_hash)
        self._contract_client.validate_tx_receipt(tx_receipt)
        if __debug__:
            retrieved_txn_hash = self._contract_client.get_mark_data_retrieved_event(dp_round, self._contract_client.address, client)['transactionHash']
            assert txn_hash == retrieved_txn_hash

    @log(_logger, logging.INFO, log_return=False)
    def _mark_clients_as_retrieved(self, dp_round: int, data: Dict[EthereumAddress, EvaluationModel]) -> None:
        futures: List[concurrent.futures.Future[HexBytes]] = []
        for client in data:
            futures.append(self._executor.submit(self._mark_client_as_retrieved, dp_round, client))
        txn_hshes = wait_for_all(futures)
        futures_2: List[concurrent.futures.Future[None]] = []
        for client, txn_hash in zip(data.keys(), txn_hshes):
            futures_2.append(self._executor.submit(self._wait_for_client_retrieved, dp_round, txn_hash, client))
        wait_for_all(futures_2)

    @log(_logger, logging.INFO, log_return=False)
    def _wait_for_decryption_key_request(self, dp_round: int, txn_hash: HexBytes, client: EthereumAddress) -> None:
        tx_receipt = self._contract_client.wait_for_tx_receipt(txn_hash)
        self._contract_client.validate_tx_receipt(tx_receipt)
        if __debug__:
            decryption_key_requests = self._contract_client.get_decryption_key_requests(dp_round, client)
            found_request = False
            for request_auditor, request_client in decryption_key_requests:
                assert request_client == client, "invariant violation"
                if request_auditor == self._contract_client.address:
                    found_request = True
                    break
            assert found_request, "my request not found"

    @log(_logger, logging.INFO, log_return=False)
    def _request_decryption_keys(self, dp_round: int, clients_to_ipfs_folders: Mapping[EthereumAddress, cid.CIDv0], data: Dict[EthereumAddress, EvaluationModel]) -> None:
        futures: List[concurrent.futures.Future[HexBytes]] = []
        for client in clients_to_ipfs_folders:
            if client not in data:
                futures.append(self._executor.submit(self._contract_client.request_decryption_key, dp_round, client))
        futures_2: List[concurrent.futures.Future[None]] = []
        txn_hshes = wait_for_all(futures)
        for client, txn_hash in zip(clients_to_ipfs_folders.keys(), txn_hshes):
            futures_2.append(self._executor.submit(self._wait_for_decryption_key_request, dp_round, txn_hash, client))
        wait_for_all(futures_2)

    @log(_logger, logging.INFO, log_return=False)
    def _retrieve_decryption_keys_and_data(self, dp_round: int, clients_to_ipfs_folders: Mapping[EthereumAddress, cid.CIDv0], data: Dict[EthereumAddress, EvaluationModel]) -> None:
        decryption_key_responses = self._contract_client.get_provided_decryption_keys(dp_round, auditor=self._contract_client.address)
        futures: List[concurrent.futures.Future[None]] = []
        for decryption_key_response in decryption_key_responses:
            client = decryption_key_response.client
            for client, ipfs_folder in clients_to_ipfs_folders.items():
                futures.append(self._executor.submit(self._retrieve_client_data, dp_round, client, ipfs_folder, data, decryption_key_response.provider, decryption_key_response.decryption_key_ipfs_address))
        # Safe to ignore exceptions. Ideally limit to client errors. It is expected for the called method to throw if there is an issue with another client's submission
        concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_EXCEPTION)  # TODO remove first exception

    @log(_logger, logging.INFO, log_return=False)
    def data_retrieval(self, dp_round: int) -> Dict[EthereumAddress, EvaluationModel]:
        data_retrieval_end_block = self._contract_client.get_scoring_start_block(dp_round)
        client_to_ipfs_folders = self._contract_client.get_data_ipfs_addresses(dp_round)
        data: Dict[EthereumAddress, EvaluationModel] = {}
        self._retrieve_data(dp_round, client_to_ipfs_folders, data)
        if self._contract_client.is_client_retrieving_data(self._contract_client.address) and self._contract_client.get_latest_block_number() < data_retrieval_end_block:
            while len(data) < len(client_to_ipfs_folders) and self._contract_client.get_latest_block_number() < data_retrieval_end_block:
                self._logger.info("Waiting for %d keys", len(client_to_ipfs_folders) - len(data))
                self._request_decryption_keys(dp_round, client_to_ipfs_folders, data)
                self._retrieve_decryption_keys_and_data(dp_round, client_to_ipfs_folders, data)
                if self._contract_client.get_latest_block_number() >= self._contract_client.get_data_retrieval_submission_start_block(dp_round):
                    self._boot_inactive_clients(dp_round)
                    self._mark_clients_as_retrieved(dp_round, data)
                time.sleep(1.0)  # sleeping to prevent hammering web3
            self._logger.info("Waiting for data retrieval submission start block")
            self._contract_client.wait_for_block(BlockNumber(self._contract_client.get_data_retrieval_submission_start_block(dp_round)))
            if self._contract_client.get_latest_block_number() < data_retrieval_end_block:
                self._boot_inactive_clients(dp_round)
                self._mark_clients_as_retrieved(dp_round, data)
            self._logger.info("Successfully retrieved %d of %d keys", len(data), len(client_to_ipfs_folders))
        self._logger.info("Waiting for data retrieval end block")
        self._contract_client.wait_for_block(data_retrieval_end_block)
        assert self._contract_client.get_latest_block_number() >= data_retrieval_end_block, "block not yet at end"
        return data
