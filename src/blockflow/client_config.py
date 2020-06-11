from dataclasses import dataclass
from typing import Optional

from dataclasses_json.api import dataclass_json

@dataclass_json
@dataclass(frozen=True)
class ClientConfig:
    encryption_keys_folder: str
    private_key: str
    upload_buffer_blocks: int
    retrieved_data_folder: Optional[str] = None  # defaults to a temp directory
    max_threads: Optional[int] = None # defaults to min(32, cpu_count + 4)
    max_web3_connections: Optional[int] = None # defaults to max_threads
    enable_encryption: bool = True
    # whether to tally on scores submitted as an auditor, and to commit only your model's score. if all clients have this setting enabled, and one fails to tally their required scores, then the smart contract will fail.
    # should be True only for testing with non-malicious clients. Saves hammering the smart contract with failing transactions
    tally_self_only: bool = False
