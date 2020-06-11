from typing import Dict, Hashable
import threading

_LOCKS: Dict[Hashable, threading.Lock] = {}
_INSERTION_LOCK = threading.Lock()  # this lock should be process-local, thread-global

def get_global_lock(key: Hashable) -> threading.Lock:
    if key in _LOCKS:
        return _LOCKS[key]
    with _INSERTION_LOCK:
        if key not in _LOCKS:
            _LOCKS[key] = threading.Lock()
        return _LOCKS[key]
