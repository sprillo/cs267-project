import hashlib
import logging
import os
import sys
from inspect import signature
from typing import List


def _init_logger():
    logger = logging.getLogger("caching")
    logger.setLevel(logging.INFO)
    fmt_str = "[%(asctime)s] - %(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(fmt_str)

    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(formatter)
    logger.addHandler(consoleHandler)


_init_logger()
logger = logging.getLogger("caching")

_CACHE_DIR = None
_HASH = True


def set_cache_dir(cache_dir: str):
    logger.info(f"Setting cache directory to: {cache_dir}")
    global _CACHE_DIR
    _CACHE_DIR = cache_dir


def get_cache_dir():
    global _CACHE_DIR
    if _CACHE_DIR is None:
        raise Exception(
            "Cache directory has not been set yet. "
            "Please set it with set_cache_dir function."
        )
    return _CACHE_DIR


def set_log_level(log_level: int):
    logger = logging.getLogger("caching")
    logger.setLevel(level=log_level)


def set_hash(hash: bool):
    logger.info(f"Setting cache to use hash: {hash}")
    global _HASH
    _HASH = hash


def get_hash():
    global _HASH
    return _HASH


class CacheUsageError(Exception):
    pass


def _hash_all(xs: List[str]) -> str:
    hashes = [hashlib.sha512(x.encode("utf-8")).hexdigest() for x in xs]
    return hashlib.sha512("".join(hashes).encode("utf-8")).hexdigest()


def _get_func_caching_dir(
    func, unhashed_args: List[str], args, kwargs, cache_dir: str, hash: bool
) -> str:
    """
    Get caching directory for the given *function call*.

    The arguments in unhashed_args are not included in the cache key.
    """
    # Get the binding
    s = signature(func)
    binding = s.bind(*args, **kwargs)
    binding.apply_defaults()

    # Compute the cache key
    if not hash:
        path = (
            [cache_dir]
            + [
                f"{func.__name__}"
            ]  # TODO: use the function name _and_ the module name? (To avoid function name collision with other modules that use the caching decorator)
            + [
                f"{key}_{val}"
                for (key, val) in binding.arguments.items()
                if key not in unhashed_args
            ]
        )
    else:
        path = (
            [cache_dir]
            + [
                f"{func.__name__}"
            ]  # TODO: use the function name _and_ the module name? (To avoid function name collision with other modules that use the caching decorator)
            + [
                _hash_all(
                    [
                        f"{key}_{val}"
                        for (key, val) in binding.arguments.items()
                        if (key not in unhashed_args)
                    ]
                )
            ]
        )
    func_caching_dir = os.path.join(*path)
    return func_caching_dir


def _validate_decorator_args(
    func,
    decorator_args: List[str],
) -> None:
    """
    Validate that the decorator arguments makes sense.

    Raises:
        CacheUsageError is any error is detected.
    """
    # Check that all arguments specified in the decorator are arguments of the
    # wrapped function - the user might have made a typo!
    func_parameters = list(signature(func).parameters.keys())
    for arg in decorator_args:
        if arg not in func_parameters:
            raise CacheUsageError(
                f"{arg} is not an argument to '{func.__name__}'. Fix the "
                f"arguments of the caching decorator."
            )

    # No argument can be repeated in the decorator
    if len(set(decorator_args)) != len(decorator_args):
        raise CacheUsageError(
            "All the function arguments specified in the caching decorator for "
            f"'{func.__name__}' should be distinct. You provided: "
            f"{decorator_args} "
        )


def _get_mode(path):
    """
    Get mode of a file (e.g. '664', '555')
    """
    return oct(os.stat(path).st_mode)[-3:]