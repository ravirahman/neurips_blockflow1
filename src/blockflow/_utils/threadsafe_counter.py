import threading
from typing import Dict, Hashable, Optional, Type
from types import TracebackType

class ThreadsafeCounter:
    def __init__(self, initial_value: int = 0) -> None:
        self._counter = initial_value
        self._lock = threading.RLock()

    def __enter__(self) -> 'ThreadsafeCounter':
        self._lock.acquire()
        return self

    def __exit__(self, exc_type: Optional[Type[BaseException]], exc_val: Optional[BaseException], exc_tb: Optional[TracebackType]) -> None:
        self._lock.release()

    def get_value(self) -> int:
        return self._counter

    def increment(self) -> None:
        with self._lock:
            self._counter += 1

    def get_value_and_increment(self) -> int:
        with self._lock:
            value = self._counter
            self._counter += 1
        return value

_COUNTERS: Dict[Hashable, ThreadsafeCounter] = {}
_INSERTION_LOCK = threading.Lock()  # this lock should be process-local, thread-global

def get_global_counter(key: Hashable, initial_value: Optional[int] = None) -> ThreadsafeCounter:
    if key in _COUNTERS:
        return _COUNTERS[key]
    with _INSERTION_LOCK:
        if key not in _COUNTERS:
            if initial_value is None:
                raise Exception("counter not initialized and initial value not provided")
            _COUNTERS[key] = ThreadsafeCounter(initial_value)
        return _COUNTERS[key]
