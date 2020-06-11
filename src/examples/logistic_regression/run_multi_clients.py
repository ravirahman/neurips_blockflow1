from multiprocessing import get_context
import logging
import os
import argparse
import warnings

from eth_utils.hexadecimal import decode_hex
from eth_utils.currency import to_wei

from blockflow.contract.contract_parameters import ContractParameters

from .._utils.deploy_contract import deploy_contract
from .._utils.web3_factory import ExampleWeb3Factory
from ..config import WEB3_URI, PRIVATE_KEYS, MAX_THREADS, MAX_WEB3_CONNECTIONS, IPFS_HTTP_PASSWORD, IPFS_HTTP_USERNAME, IPFS_URI, RESULTS_FOLDER_PATH
from .logistic_regression_runner import run_client, LogisticRegressionRunConfig

if not __debug__:
    warnings.filterwarnings('ignore')

def run_multi_clients(*, experiment_name: str, dataset_parent_folder: str, apply_dp: bool, inv_lambda: float, epsilon: float, ground_truth_dataset: str) -> None:
    logging.basicConfig(
        format='%(asctime)s %(levelname)s {%(module)s} [%(funcName)s] %(message)s'
    )
    logging.getLogger("blockflow").setLevel(logging.DEBUG if __debug__ else logging.INFO)
    logging.getLogger("examples").setLevel(logging.DEBUG if __debug__ else logging.INFO)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    client_dataset_folders = os.listdir(dataset_parent_folder)
    client_dataset_folders.sort()

    num_clients = len(client_dataset_folders)
    contract_parameters = ContractParameters(
        bond_amount=to_wei(1000, 'microether'),
        bond_reserve_amount=to_wei(200, 'microether'),
        start_block=(40+num_clients),
        dp_round_training_blockdelta=(30+4*num_clients),
        dp_round_data_retrieval_blockdelta=(20+3*num_clients),
        dp_round_scoring_blockdelta=(20+3*num_clients),
        dp_round_score_decrypting_blockdelta=(20+4*num_clients),
        submission_blockdelta=20+num_clients * 10,
        num_dp_rounds=5
    )
    web3_factory = ExampleWeb3Factory(WEB3_URI)
    address = deploy_contract(web3_factory, decode_hex(PRIVATE_KEYS[0]), contract_parameters, start_block_is_relative=True)
    ctx = get_context("spawn")
    processes = []
    for i, client_folder in enumerate(client_dataset_folders):
        run_config = LogisticRegressionRunConfig(
            results_folder_path=RESULTS_FOLDER_PATH,
            max_threads=MAX_THREADS,
            experiment_name=experiment_name,
            client_private_key=PRIVATE_KEYS[i],
            contract_address=address,
            web3_uri=WEB3_URI,
            max_web3_connections=MAX_WEB3_CONNECTIONS,
            upload_block_buffer=10,
            max_num_epochs=100,
            inv_lambda=inv_lambda,  # inverse regular strength (lambda)
            epsilon=epsilon,
            apply_dp=apply_dp,
            ground_truth_dataset=ground_truth_dataset,
            client_folder=os.path.join(dataset_parent_folder, client_folder),
            ipfs_uri=IPFS_URI,
            ipfs_http_username=IPFS_HTTP_USERNAME,
            ipfs_http_password=IPFS_HTTP_PASSWORD)
        proc = ctx.Process(target=run_client, args=(run_config, ))
        proc.start()
        processes.append(proc)
    for proc in processes:
        proc.join()

def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", choices=('kdd99', 'adult'), required=True)
    parser.add_argument("--inv_lambda", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=1e-3)
    parser.add_argument("--apply_dp", action='store_true')
    parser.add_argument("--dataset_folder", type=str, required=True)
    parser.add_argument("--num_dp_rounds", type=int, default=5)
    parser.add_argument("--ground_truth_dataset", type=str, required=True)
    args = parser.parse_args()
    run_multi_clients(experiment_name=args.exp_name,
                      dataset_parent_folder=args.dataset_folder,
                      apply_dp=args.apply_dp,
                      inv_lambda=args.inv_lambda,
                      epsilon=args.epsilon,
                      ground_truth_dataset=args.ground_truth_dataset)


if __name__ == "__main__":
    _main()
