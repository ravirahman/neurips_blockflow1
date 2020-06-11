import os
import logging
import subprocess
import json
import time

import web3
import web3.contract
import web3.exceptions

from .contract_parameters import ContractParameters
from .ethereum_address import EthereumAddress
from .wwwrapper import WWWrapper, VyperOutput

class ContractFactory:
    _logger = logging.getLogger(__name__)

    def __init__(self, combined_json_path: str = os.path.join(os.path.dirname(__file__), "..", "..", "vyper", "generated", "master.combined_json.json")) -> None:
        self._logger.info("Using made agreement at path %s", combined_json_path)
        with open(combined_json_path, "r") as f:
            combined_output = json.load(f)
        # stdout =
        # combined_output = json.loads(stdout)
        output = combined_output[next(iter(combined_output.keys()))]
        bytecode: str = output['bytecode']
        abi: str = output['abi']
        source_map: str = output['source_map']
        self._vyper_output = VyperOutput(bytecode=bytecode, abi=abi, src_map=source_map)

    @staticmethod
    def make_clean_and_make() -> None:
        subprocess.check_call(["make clean"], cwd=os.path.join(os.path.dirname(__file__), "..", "..", "vyper"))
        subprocess.check_call(["make"], cwd=os.path.join(os.path.dirname(__file__), "..", "..", "vyper"))

    def deploy(self, wwwrapper: WWWrapper, parameters: ContractParameters) -> web3.contract.Contract:
        contract_type = wwwrapper.build_contract_type(self._vyper_output)
        constructor = contract_type.constructor(parameters.num_dp_rounds,
                                                parameters.bond_amount,
                                                parameters.bond_reserve_amount,
                                                parameters.start_block,
                                                parameters.submission_blockdelta,
                                                parameters.dp_round_training_blockdelta,
                                                parameters.dp_round_data_retrieval_blockdelta,
                                                parameters.dp_round_scoring_blockdelta,
                                                parameters.dp_round_score_decrypting_blockdelta,
                                                (parameters.min_agreement_threshold_num, parameters.min_agreement_threshold_denom),
                                                (parameters.refund_fraction_num, parameters.refund_fraction_denom),
                                                web3.Web3.toChecksumAddress(parameters.client_authorizer))
        txn_hash = wwwrapper.transact(constructor)
        while True:
            try:
                tx_receipt = wwwrapper.get_transaction_receipt(txn_hash)
                break
            except web3.exceptions.TransactionNotFound:
                time.sleep(1.0)
        deployment_gas = tx_receipt['gasUsed']
        print("deployment_gas", deployment_gas)
        address: EthereumAddress = tx_receipt.contractAddress
        contract = contract_type(address=address)
        assert isinstance(contract, web3.contract.Contract)
        return contract

    def use(self, wwwrapper: WWWrapper, address: EthereumAddress) -> web3.contract.Contract:
        contract_type = wwwrapper.build_contract_type(self._vyper_output)
        contract = contract_type(address=address)
        assert isinstance(contract, web3.contract.Contract)
        return contract
