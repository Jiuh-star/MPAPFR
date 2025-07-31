from __future__ import annotations

import contextlib
import pathlib
import time
from typing import IO, Any, Generator, Literal

import filelock
import torch
from loguru import logger

__all__ = ["dump", "load", "atomic_open", "set_io_policy", "get_io_policy"]


_io_policy: Literal["real", "virtual"] = "real"
_lock = filelock.FileLock("atomic_io.lock", is_singleton=True)
_storage = {}


def set_io_policy(policy: Literal["real", "virtual"]):
    """
    Set the I/O policy, which determines whether to store data in memory or on disk.

    :param policy: The policy. If the policy is "memory_first", data will be dumped to disk.
    If the policy is "speed_first", data will be stored in memory.
    """
    global _io_policy

    if policy not in ["real", "virtual"]:
        raise ValueError(f"Invalid I/O policy: {policy}")

    with _lock:
        _io_policy = policy


def get_io_policy():
    """
    Get the I/O policy.

    :return: The policy.
    """
    return _io_policy


def _real_dump(obj: Any, filename: pathlib.Path, *, replace: bool = False):
    temp_file = filename.with_suffix(".tmp")
    lock = filelock.FileLock(str(filename) + ".lock", timeout=60, is_singleton=True)

    with lock:
        temp_file.unlink(missing_ok=True)
        torch.save(obj, temp_file)

        for i in range(5):  # to fix randomly occurred permission error
            try:
                if replace:
                    temp_file.replace(filename)
                else:
                    temp_file.rename(filename)
                break
            except PermissionError as e:
                logger.warning(f"Permission error: {e}")
                time.sleep(2 * (i + 1))
        else:
            logger.error(f"Unfixable permission error: {filename}")
            breakpoint()


def _real_load(filename: pathlib.Path, *, raise_error: bool = True, **kwargs) -> Any:
    cm = contextlib.nullcontext() if raise_error else contextlib.suppress(Exception)

    with cm:
        return torch.load(filename, weights_only=False, **kwargs)


def _virtual_dump(obj: Any, filename: pathlib.Path, *, replace: bool = False):
    with _lock:
        if filename in _storage and not replace:
            raise FileExistsError(f"Same filename already exists: {filename}")
        _storage[filename] = obj


def _virtual_load(filename: pathlib.Path, *, raise_error: bool = True, **kwargs) -> Any:
    with _lock:
        if filename not in _storage:
            obj = _memory_first_load(filename, raise_error=raise_error, **kwargs)
            return obj
        return _storage[filename]


def dump(obj: Any, filename: str | pathlib.Path, *, replace: bool = False):
    """
    Dump object `obj` into file.

    :param obj: The object to dump.
    :param filename: The filename.
    :param replace: Replace the file if it exists.

    :raise FileExistsError: If file exists and `replace` is False.
    """
    filename = pathlib.Path(filename)
    if _io_policy == "virtual":
        _virtual_dump(obj, filename, replace=replace)
    else:
        _real_dump(obj, filename, replace=replace)


def load(filename: str | pathlib.Path, *, raise_error: bool = True, **kwargs) -> Any:
    """
    Load object from file.

    :param filename: The filename.
    :param raise_error: Raise error if file not exists.
    :return: The object.
    """
    filename = pathlib.Path(filename)
    if _io_policy == "virtual":
        return _virtual_load(filename, raise_error=raise_error, **kwargs)
    else:
        return _real_load(filename, raise_error=raise_error, **kwargs)


@contextlib.contextmanager
def atomic_open(filename: str | pathlib.Path, mode: str = "r") -> Generator[IO, None, None]:
    """
    Open file in atomic mode (write to temp file first, then rename to target file).
    This function supports multiple processes writing to the same file.

    :param filename: The filename.
    :param mode: The mode.
    :return: The file object.
    """
    filename = pathlib.Path(filename)
    temp_file = filename.with_suffix(".tmp")
    lock = filelock.FileLock(str(filename) + ".lock", timeout=60, is_singleton=True)

    with lock:
        with open(temp_file, mode) as f:
            yield f

        temp_file.replace(filename)
