import logging
import os
import shutil
from typing import NamedTuple, Optional

import tensorboardX
import eth_account
from eth_utils.address import to_checksum_address

from blockflow.client_config import ClientConfig
from blockflow.experiment import Experiment
from blockflow.statistics.tensorboardx import TensorboardXRecorder
from blockflow.ipfs_client import IPFSClient
from ..dp.dp_base import DPBase
from ..dp.laplace_output import LaplaceOutputDP
from .model import LogisticRegressionModelFactory, LogisticRegressionEvaluationModelParams, LogisticRegressionTrainingModelParams

from .._utils.web3_factory import ExampleWeb3Factory
from .adult.dataset import ADULT_OUTPUT_CLASSES, ADULT_NUM_COEFS
from .kdd99.dataset import KDD99_OUTPUT_CLASSES, KDD99_NUM_COEFS
from .dataset import LogisticRegressionDataset
from .hooks import LRHooks

class LogisticRegressionRunConfig(NamedTuple):
    experiment_name: str
    client_private_key: str
    contract_address: str
    results_folder_path: str
    max_threads: int
    max_web3_connections: int
    upload_block_buffer: int
    web3_uri: str
    max_num_epochs: int
    inv_lambda: float  # inverse regular strength (lambda)
    epsilon: float
    apply_dp: bool
    client_folder: str
    ground_truth_dataset: str
    ipfs_uri: str
    ipfs_http_username: str
    ipfs_http_password: str

def run_client(run_config: LogisticRegressionRunConfig) -> None:
    client_address = eth_account.Account.from_key(run_config.client_private_key).address  # pylint: disable=no-value-for-parameter
    client_folder = os.path.dirname(os.path.join(run_config.client_folder, "test"))
    client_str = f"client_{client_address}"
    client_folder_basename = os.path.basename(client_folder)
    run_data_folder = os.path.dirname(client_folder)
    if run_config.experiment_name == "kdd99":
        output_classes = KDD99_OUTPUT_CLASSES
        num_coefs = KDD99_NUM_COEFS
    elif run_config.experiment_name == "adult":
        output_classes = ADULT_OUTPUT_CLASSES
        num_coefs = ADULT_NUM_COEFS
    else:
        raise Exception("unknown experiment")

    safe_name = os.path.join(run_data_folder.replace("/", "_").replace("\\", "_"), "inv_lambda-" + str(run_config.inv_lambda) + "_epsilon-" + str(run_config.epsilon) + "_apply_dp-" + str(run_config.apply_dp))
    tensorboard_folder = os.path.join(run_config.results_folder_path, "tensorboard", run_config.experiment_name, safe_name, client_str)
    shutil.rmtree(tensorboard_folder, ignore_errors=True)
    os.makedirs(tensorboard_folder)

    run_folder = os.path.join(run_config.results_folder_path, run_config.experiment_name, safe_name, client_str)
    os.makedirs(run_folder, exist_ok=True)

    model_checkpoints = os.path.join(run_folder, "model_checkpoints")
    shutil.rmtree(model_checkpoints, ignore_errors=True)
    os.makedirs(model_checkpoints)

    keys_folder = os.path.join(run_folder, "keys", client_str)
    shutil.rmtree(keys_folder, ignore_errors=True)
    os.makedirs(keys_folder)

    logging.basicConfig(
        filemode='w+',
        format='%(asctime)s %(levelname)s %(name)s %(module)s %(funcName)s %(message)s',
        filename=os.path.join(run_folder, f"logs.log"),
    )
    logging.getLogger("blockflow").setLevel(logging.DEBUG if __debug__ else logging.INFO)
    logging.getLogger("examples").setLevel(logging.DEBUG if __debug__ else logging.INFO)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG if __debug__ else logging.INFO)

    training_dataset_path = os.path.join(run_config.client_folder, "train.dat")
    validation_dataset_path = os.path.join(run_config.client_folder, "validation.dat")
    score_dataset_path = os.path.join(run_config.client_folder, "score.dat")  # when evaluating others' models, use all ur data

    training_dataset = LogisticRegressionDataset.load(training_dataset_path)
    ground_truth_dataset = LogisticRegressionDataset.load(run_config.ground_truth_dataset)

    g = 1.0  # g = 1.0 for logistic regressions
    if run_config.apply_dp:
        dp: DPBase = LaplaceOutputDP(inv_reg_term=run_config.inv_lambda, epsilon=run_config.epsilon)
    else:
        dp = DPBase()
    training_model_parameters = LogisticRegressionTrainingModelParams(
        training_dataset=training_dataset,
        evaluation_params=LogisticRegressionEvaluationModelParams(
            output_classes=output_classes,
            num_coefs=num_coefs,
            validation_dataset=LogisticRegressionDataset.load(validation_dataset_path),
            sklearn_lr_kwargs={"C": run_config.inv_lambda}
        ),
        max_num_epochs=run_config.max_num_epochs,
        checkpoint_folder=model_checkpoints,
        dp=dp)
    score_model_params = LogisticRegressionEvaluationModelParams(
        output_classes=output_classes,
        num_coefs=num_coefs,
        validation_dataset=LogisticRegressionDataset.load(score_dataset_path),
        sklearn_lr_kwargs={"C": run_config.inv_lambda}
    )
    with tensorboardX.SummaryWriter(tensorboard_folder) as summary_writer:
        client_config = ClientConfig(encryption_keys_folder=keys_folder,
                                     private_key=run_config.client_private_key,
                                     max_threads=run_config.max_threads,
                                     max_web3_connections=run_config.max_web3_connections,
                                     upload_buffer_blocks=run_config.upload_block_buffer,
                                     enable_encryption=False,  # encryption is very slow in python
                                     tally_self_only=True)  # don't need the redundancy for evaluation
        chunk_size = 512*1024*1024
        ipfs_client = IPFSClient(chunk_size, run_config.ipfs_uri, username=run_config.ipfs_http_username, password=run_config.ipfs_http_password)
        config_dict = run_config._asdict()
        config_dict.pop("client_private_key")
        config_dict.pop("web3_uri")
        config_dict.pop("ipfs_uri")
        config_dict.pop("ipfs_http_username")
        config_dict.pop("ipfs_http_password")
        for name, value in config_dict.items():
            summary_writer.add_text(name, str(value), global_step=0)
        hooks = LRHooks(ground_truth_dataset)
        with ipfs_client:
            statistics_recorder = TensorboardXRecorder(summary_writer)
            experiment = Experiment(model_factory=LogisticRegressionModelFactory(training_model_parameters, score_model_params),
                                    web3_factory=ExampleWeb3Factory(run_config.web3_uri),
                                    address=to_checksum_address(run_config.contract_address),
                                    ipfs_client=ipfs_client,
                                    config=client_config,
                                    statistics_recorder=statistics_recorder,
                                    hooks=hooks)
            with experiment:
                logger.info("starting experiment for client %s", client_address)
                experiment.run()
