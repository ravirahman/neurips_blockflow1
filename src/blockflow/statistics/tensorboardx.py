from typing import Dict, Union
from dataclasses import asdict

from tensorboardX import SummaryWriter

from ..contract.ethereum_address import EthereumAddress
from ..contract.contract_parameters import ContractParameters

from .base import BaseRecorder

class TensorboardXRecorder(BaseRecorder):
    def __init__(self, summary_writer: SummaryWriter, metric_name_prefix: str = '/') -> None:
        self._summary_writer = summary_writer
        assert metric_name_prefix.endswith('/'), "metric_name_prefix should end with a /"
        self._metric_name_prefix = metric_name_prefix

    def record_scalar(self, dp_round: int, metric_name: str, value: Union[int, float]) -> None:
        self._summary_writer.add_scalar(self._metric_name_prefix + metric_name, value, global_step=dp_round)

    def record_scalars(self, dp_round: int, metric_name: str, client_to_value: Dict[EthereumAddress, float]) -> None:
        self._summary_writer.add_scalars(self._metric_name_prefix + metric_name, client_to_value, global_step=dp_round)

    def record_text(self, dp_round: int, name: str, value: str) -> None:
        self._summary_writer.add_text(name, value, global_step=dp_round)

    def record_parameters(self, dp_round: int, contract_parameters: ContractParameters, metrics: Dict[str, float]) -> None:
        hparams: Dict[str, Union[bool, str, float, int, None]] = {}
        hparams.update(asdict(contract_parameters))
        metrics_prefixed: Dict[str, float] = {}
        for metric, value in metrics.items():
            metrics_prefixed[self._metric_name_prefix + "hparams/" + metric] = value
        self._summary_writer.add_hparams(hparams, metrics, global_step=dp_round)
