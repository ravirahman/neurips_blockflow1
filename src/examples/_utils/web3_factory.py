import web3
from eth_typing.ethpm import URI
from web3.gas_strategies.rpc import rpc_gas_price_strategy
from web3.middleware.geth_poa import geth_poa_middleware
from web3.providers.auto import load_provider_from_uri

from blockflow.contract.web3_factory import Web3Factory

class ExampleWeb3Factory(Web3Factory):
    def __init__(self, uri: str) -> None:
        self._uri = URI(uri)

    def build(self) -> web3.Web3:
        web3_provider = load_provider_from_uri(self._uri)
        w3 = web3.Web3(web3_provider)
        w3.middleware_onion.inject(geth_poa_middleware, layer=0)
        w3.eth.setGasPriceStrategy(rpc_gas_price_strategy)  # type: ignore
        return w3
