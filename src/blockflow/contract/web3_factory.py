from abc import ABC, abstractmethod

import web3

class Web3Factory(ABC):
    @abstractmethod
    def build(self) -> web3.Web3:
        pass
