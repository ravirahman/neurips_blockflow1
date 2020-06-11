import logging
import os
import shutil
from typing import NamedTuple

import tensorboardX

from blockflow.contract.ethereum_address import EthereumAddress
from blockflow.client_config import ClientConfig
from blockflow.experiment import Experiment
from blockflow.statistics.tensorboardx import TensorboardXRecorder
from blockflow.ipfs_client import IPFSClient

from ...config import WEB3_URI, IPFS_URI, IPFS_HTTP_USERNAME, IPFS_HTTP_PASSWORD, PRIVATE_KEYS, MAX_THREADS, MAX_WEB3_CONNECTIONS
from ..._utils.web3_factory import ExampleWeb3Factory

from .dataset import MyDatasetFactory
from .model import MyModelFactory


class MyDatasetParams(NamedTuple):
    validation_noise_factor: float = 0.0
    training_noise_factor: float = 0.0
    training_offset: float = 0.0
    validation_offset: float = 0.0
    validation_correct_factor: float = 1.0
    training_correct_factor: float = 1.0
    training_dataset_size: int = 10


def run_client(run_name: str, client_i: int, address: EthereumAddress, logging_level: int, dataset_params: MyDatasetParams) -> None:
    client_i_str = f"client_{client_i}"
    example_folder = os.path.join("runs", "linear_regression")
    tensorboard_folder = os.path.join("runs", "tensorboard", "linear_regression", run_name, client_i_str)
    run_folder = os.path.join(example_folder, run_name)
    logs_folder = os.path.join(run_folder, "logs")
    keys_folder = os.path.join(run_folder, "keys", client_i_str)
    shutil.rmtree(tensorboard_folder, ignore_errors=True)
    os.makedirs(tensorboard_folder)
    os.makedirs(logs_folder, exist_ok=True)
    shutil.rmtree(keys_folder, ignore_errors=True)
    os.makedirs(keys_folder)
    logging.basicConfig(
        filemode='w+',
        filename=os.path.join(logs_folder, f"{client_i_str}.log"),
    )
    logging.getLogger("blockflow").setLevel(logging_level)
    logging.getLogger("examples").setLevel(logging_level)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging_level)
    with tensorboardX.SummaryWriter(tensorboard_folder) as summary_writer:
        private_key = PRIVATE_KEYS[client_i]
        client_config = ClientConfig(encryption_keys_folder=keys_folder, private_key=private_key, max_web3_connections=MAX_WEB3_CONNECTIONS, max_threads=MAX_THREADS, upload_buffer_blocks=5)
        chunk_size = 512*1024*1024
        ipfs_client = IPFSClient(chunk_size, IPFS_URI, username=IPFS_HTTP_USERNAME, password=IPFS_HTTP_PASSWORD)
        validation_dataset_factory = MyDatasetFactory(offset=dataset_params.validation_offset, correct_factor=dataset_params.validation_correct_factor, noise_factor=dataset_params.validation_noise_factor)
        validation_dataset = validation_dataset_factory.generate(100)
        training_dataset = MyDatasetFactory(correct_factor=dataset_params.training_correct_factor, offset=dataset_params.training_offset, noise_factor=dataset_params.training_noise_factor).generate(dataset_params.training_dataset_size)
        statistics_recorder = TensorboardXRecorder(summary_writer)
        summary_writer.add_scalars("dataset_params", dataset_params._asdict(), 0)
        experiment = Experiment(model_factory=MyModelFactory(training_dataset, validation_dataset),
                                web3_factory=ExampleWeb3Factory(WEB3_URI),
                                address=address,
                                ipfs_client=ipfs_client,
                                config=client_config,
                                statistics_recorder=statistics_recorder)
        with experiment as exp:
            logger.info("starting experiment for client_i %d", client_i)
            exp.run()
