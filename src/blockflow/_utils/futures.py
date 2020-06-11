from concurrent.futures import Future
from typing import Sequence, TypeVar, List

T = TypeVar('T')

def wait_for_all(futs: 'Sequence[Future[T]]') -> Sequence[T]:
    results: List[T] = []
    for fut in futs:
        results.append(fut.result())
    return results
