from typing import Optional, Type
from types import TracebackType
import os
import logging

import cid
import ipfshttpclient

from ._utils.log_wrapper import log

class AlreadyConnectedError(Exception):
    pass

class IPFSClient:
    _logger = logging.getLogger(__name__)

    def __init__(self, chunk_size: int, *args: object, **kwargs: object) -> None:
        self._chunk_size = chunk_size
        if "session" not in kwargs:
            kwargs["session"] = True
        self._client = ipfshttpclient.connect(*args, **kwargs)

    def __enter__(self) -> 'IPFSClient':
        return self

    @log(_logger, logging.INFO)
    def upload(self, filename: str) -> cid.CIDv0:
        result = self._client.add(filename, recursive=True)
        if isinstance(result, dict):
            return cid.make_cid(result['Hash'])
        assert isinstance(result, list)
        for info in result:
            if info['Name'] == os.path.basename(filename):
                return cid.make_cid(info['Hash'])
        raise Exception("invariant violation")

    @log(_logger, logging.INFO)
    def download_to_file(self, content_path: str, destination_name: str) -> None:
        with open(destination_name, "wb+") as f:  # using a tcp pool for this download
            offset = 0
            while True:
                content = self._client.cat(content_path, offset, self._chunk_size)
                if len(content) == 0:
                    break
                f.write(content)
                offset += self._chunk_size

    @log(_logger, logging.DEBUG)
    def download(self, content_id: cid.CIDv0) -> bytes:
        ans: bytes = self._client.cat(content_id)
        return ans

    def close(self) -> None:
        self._client.close()

    def __exit__(self, exc_type: Optional[Type[BaseException]], exc_val: Optional[BaseException], exc_tb: Optional[TracebackType]) -> None:
        self.close()
