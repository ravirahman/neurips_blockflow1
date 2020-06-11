from typing import Type, Optional, Tuple, Set
from types import TracebackType
import queue
import threading

import web3.contract

from .wwwrapper import WWWrapper
from .web3_factory import Web3Factory
from .contract_factory import ContractFactory
from .ethereum_address import EthereumAddress

class PoolItem:
    def __init__(self, pool: 'ContractWWWrapperPool', contract: web3.contract.Contract, wwwrapper: WWWrapper) -> None:
        self._pool = pool
        self._contract = contract
        self._wwwrapper = wwwrapper

    @property
    def contract(self) -> web3.contract.Contract:
        return self._contract

    @property
    def wwwrapper(self) -> WWWrapper:
        return self._wwwrapper

    def __enter__(self) -> Tuple[web3.contract.Contract, WWWrapper]:
        return self._contract, self._wwwrapper

    def __exit__(self, exc_type: Optional[Type[BaseException]], exc_val: Optional[BaseException], exc_tb: Optional[TracebackType]) -> None:
        self._pool.replace(self)

class ContractWWWrapperPool:
    def __init__(self, pool_size: int, client_private_key: bytes, contract_address: EthereumAddress, web3_factory: Web3Factory, contract_factory: Optional[ContractFactory] = None) -> None:
        self._pool: queue.LifoQueue[PoolItem] = queue.LifoQueue(pool_size)
        if __debug__:
            self._threads_with_pool_items: Set[int] = set()
            self._threads_with_pool_items_lock = threading.Lock()
        if contract_factory is None:
            contract_factory = ContractFactory()
        for _ in range(pool_size):
            w3 = web3_factory.build()
            wwwrapper = WWWrapper(client_private_key, w3)
            contract = contract_factory.use(wwwrapper, contract_address)
            self._pool.put_nowait(PoolItem(self, contract, wwwrapper))

    def get(self) -> PoolItem:
        thread_id = threading.get_ident()
        if __debug__:
            assert thread_id not in self._threads_with_pool_items, "thread already has a poolitem; cannot have two"
            with self._threads_with_pool_items_lock:
                self._threads_with_pool_items.add(thread_id)
        return self._pool.get()

    def replace(self, poolitem: PoolItem) -> None:
        self._pool.put_nowait(poolitem)
        if __debug__:
            thread_id = threading.get_ident()
            with self._threads_with_pool_items_lock:
                self._threads_with_pool_items.remove(thread_id)
