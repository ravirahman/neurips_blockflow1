from multiprocessing import get_context
import logging

from eth_utils.hexadecimal import decode_hex

from blockflow.contract.contract_parameters import ContractParameters

from ..._utils.deploy_contract import deploy_contract
from ..._utils.web3_factory import ExampleWeb3Factory
from ...config import WEB3_URI, PRIVATE_KEYS
from .simple_linear_regression_runner import run_client, MyDatasetParams

def _main() -> None:
    logging.basicConfig()
    logging.getLogger("blockflow").setLevel(logging.INFO)
    logging.getLogger("examples").setLevel(logging.INFO)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    num_clients = 3
    contract_parameters = ContractParameters(
        bond_amount=100,
        bond_reserve_amount=25,
        start_block=5,
        dp_round_training_blockdelta=10,
        dp_round_data_retrieval_blockdelta=4,
        dp_round_scoring_blockdelta=4,
        dp_round_score_decrypting_blockdelta=4,
        num_dp_rounds=2
    )
    web3_factory = ExampleWeb3Factory(WEB3_URI)
    contract_parameters = ContractParameters(
        bond_amount=100,
        bond_reserve_amount=25,
        start_block=10,
        dp_round_training_blockdelta=15,
        dp_round_data_retrieval_blockdelta=10,
        dp_round_scoring_blockdelta=10,
        dp_round_score_decrypting_blockdelta=10,
        num_dp_rounds=3
    )
    address = deploy_contract(web3_factory, decode_hex(PRIVATE_KEYS[0]), contract_parameters, start_block_is_relative=True)
    ctx = get_context("spawn")
    processes = []
    for i in range(num_clients):
        proc = ctx.Process(target=run_client, args=(f"multi_honest", i, address, logging.DEBUG, MyDatasetParams()))
        proc.start()
        processes.append(proc)
    for proc in processes:
        proc.join()

if __name__ == "__main__":
    _main()
