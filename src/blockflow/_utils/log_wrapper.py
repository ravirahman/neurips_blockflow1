import logging
from typing import Any, Callable, TypeVar
from functools import wraps

from .threadsafe_counter import ThreadsafeCounter

F = TypeVar('F', bound=Callable[..., Any])  # type: ignore

def log(logger: logging.Logger, level: int, *, log_args: bool = True, log_return: bool = True) -> Callable[[F], F]:  # type: ignore
    def decorator(func: F) -> F:  # type: ignore
        def state_wrapper(counter: ThreadsafeCounter) -> F:  # type: ignore
            @wraps(func)
            def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:  # type: ignore
                invocation = counter.get_value_and_increment()
                if log_args:
                    logger.log(level, "Calling function(%s), invocation(%d) with args(%s), kwargs(%s)", func.__name__, invocation, str(args), str(kwargs))
                else:
                    logger.log(level, "Calling function(%s), invocation(%d)", func.__name__, invocation)
                try:
                    answer = func(self, *args, **kwargs)
                except Exception as ex:
                    logger.log(level, "Finished EXCEPTION function(%s), invocation (%d) with exception type(%s), message(%s)", func.__name__, invocation, type(ex), str(ex))
                    raise ex
                else:
                    if log_return:
                        logger.log(level, "Finished function(%s), invocation(%d) with result(%s)", func.__name__, invocation, str(answer))
                    else:
                        logger.log(level, "Finished function(%s), invocation(%d)", func.__name__, invocation)
                    return answer
            return wrapped  # type: ignore
        return state_wrapper(ThreadsafeCounter())
    return decorator
