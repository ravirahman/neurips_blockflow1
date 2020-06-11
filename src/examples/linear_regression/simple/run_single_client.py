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
    address = deploy_contract(web3_factory, decode_hex(PRIVATE_KEYS[0]), contract_parameters, start_block_is_relative=True)
    run_client("debug_single_client", 0, address, logging.DEBUG, MyDatasetParams())

if __name__ == "__main__":
    _main()
