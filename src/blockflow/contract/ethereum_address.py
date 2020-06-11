from eth_utils.address import to_checksum_address
from eth_typing.evm import ChecksumAddress

EthereumAddress = ChecksumAddress

ZERO_ADDRESS = to_checksum_address("0x0000000000000000000000000000000000000000")
