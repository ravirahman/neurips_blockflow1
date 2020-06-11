import logging
import sys
import warnings

from eth_utils.hexadecimal import decode_hex
from eth_utils.currency import to_wei

from blockflow.contract.contract_parameters import ContractParameters

from .._utils.web3_factory import ExampleWeb3Factory
from .._utils.deploy_contract import deploy_contract
from .logistic_regression_runner import LogisticRegressionRunConfig, run_client
from ..config import WEB3_URI, IPFS_HTTP_PASSWORD, IPFS_HTTP_USERNAME, IPFS_URI, MAX_THREADS, RESULTS_FOLDER_PATH, MAX_WEB3_CONNECTIONS, PRIVATE_KEYS

if not __debug__:
    warnings.filterwarnings('ignore')

def run_single_client(experiment_name: str, client_folder: str, ground_truth_dataset: str) -> None:
    logging.basicConfig()
    logging.getLogger("blockflow").setLevel(logging.DEBUG if __debug__ else logging.INFO)
    logging.getLogger("examples").setLevel(logging.DEBUG if __debug__ else logging.INFO)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    contract_parameters = ContractParameters(
        bond_amount=to_wei(1000, 'microether'),
        bond_reserve_amount=to_wei(200, 'microether'),
        start_block=12,
        dp_round_training_blockdelta=22,
        dp_round_data_retrieval_blockdelta=10,
        dp_round_scoring_blockdelta=10,
        dp_round_score_decrypting_blockdelta=10,
        submission_blockdelta=3000,
        num_dp_rounds=5
    )

    web3_factory = ExampleWeb3Factory(WEB3_URI)
    address = deploy_contract(web3_factory, decode_hex(PRIVATE_KEYS[0]), contract_parameters, start_block_is_relative=True)
    run_config = LogisticRegressionRunConfig(
        results_folder_path=RESULTS_FOLDER_PATH,
        max_threads=MAX_THREADS,
        experiment_name=experiment_name,
        client_private_key=PRIVATE_KEYS[0],
        contract_address=address,
        web3_uri=WEB3_URI,
        max_web3_connections=MAX_WEB3_CONNECTIONS,
        upload_block_buffer=10,
        max_num_epochs=100,
        inv_lambda=1.0,  # inverse regular strength (lambda)
        epsilon=1e-3,
        apply_dp=True,
        client_folder=client_folder,
        ground_truth_dataset=ground_truth_dataset,
        ipfs_uri=IPFS_URI,
        ipfs_http_username=IPFS_HTTP_USERNAME,
        ipfs_http_password=IPFS_HTTP_PASSWORD
    )
    run_client(run_config)

if __name__ == "__main__":
    run_single_client(sys.argv[1], sys.argv[2], sys.argv[3])
