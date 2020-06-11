from abc import ABC, abstractmethod
from typing import Dict, Type, Optional, Union
from types import TracebackType

from ..contract.ethereum_address import EthereumAddress
from ..contract.contract_parameters import ContractParameters

class BaseRecorder(ABC):
    @abstractmethod
    def record_scalar(self, dp_round: int, metric_name: str, value: Union[int, float]) -> None:
        pass

    @abstractmethod
    def record_scalars(self, dp_round: int, metric_name: str, client_to_value: Dict[EthereumAddress, float]) -> None:
        pass

    @abstractmethod
    def record_parameters(self, dp_round: int, contract_parameters: ContractParameters, metrics: Dict[str, float]) -> None:
        pass

    @abstractmethod
    def record_text(self, dp_round: int, name: str, value: str) -> None:
        pass

    def __enter__(self) -> 'BaseRecorder':
        return self

    def __exit__(self, exc_type: Optional[Type[BaseException]], exc_val: Optional[BaseException], exc_tb: Optional[TracebackType]) -> None:
        self.close()

    def close(self) -> None:
        pass
    