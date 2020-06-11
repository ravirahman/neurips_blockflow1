from dataclasses import dataclass

from dataclasses_json.api import dataclass_json

from .ethereum_address import ZERO_ADDRESS

@dataclass_json
@dataclass(frozen=True)
class ContractParameters:
    num_dp_rounds: int
    bond_amount: int
    bond_reserve_amount: int
    start_block: int
    dp_round_training_blockdelta: int
    dp_round_data_retrieval_blockdelta: int
    dp_round_scoring_blockdelta: int
    dp_round_score_decrypting_blockdelta: int
    min_agreement_threshold_num: int = 2
    min_agreement_threshold_denom: int = 3
    refund_fraction_num: int = 1
    refund_fraction_denom: int = 2
    submission_blockdelta: int = 1024
    client_authorizer: str = str(ZERO_ADDRESS)
