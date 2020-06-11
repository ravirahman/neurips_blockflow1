import logging
import dataclasses

from blockflow.contract.contract_factory import ContractFactory
from blockflow.contract.contract_parameters import ContractParameters
from blockflow.contract.ethereum_address import EthereumAddress
from blockflow.contract.wwwrapper import WWWrapper
from blockflow.contract.web3_factory import Web3Factory

def deploy_contract(web3_factory: Web3Factory, deployment_private_key: bytes, contract_parameters: ContractParameters, start_block_is_relative: bool) -> EthereumAddress:
    logger = logging.getLogger(__name__)
    contract_factory = ContractFactory()
    w3 = web3_factory.build()
    wwwrapper = WWWrapper(deployment_private_key, w3)
    if start_block_is_relative:
        contract_parameters = dataclasses.replace(contract_parameters, start_block=contract_parameters.start_block + wwwrapper.get_latest_block_number())
    contract = contract_factory.deploy(wwwrapper, contract_parameters)
    logger.info("deployed contract to %s", contract.address)
    return EthereumAddress(contract.address)
