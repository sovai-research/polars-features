"""Zero-dependency progress-bar shim.

``tqdm`` is an optional extra, not a core dependency. Code that wants a progress
bar imports :func:`progress` / :func:`trange` from here: if ``tqdm`` is installed
you get a real bar, otherwise you get a transparent pass-through with identical
iteration semantics and no output. This keeps the mandatory footprint to
numpy + polars while preserving the nice UX when ``tqdm`` is present.

No third-party imports at module import time — ``tqdm`` is imported lazily inside
the functions only when it is actually available.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any, TypeVar

_T = TypeVar("_T")


def _tqdm_or_none() -> Any:
    try:
        import tqdm as _tqdm  # noqa: PLC0415  (intentional lazy import)

        return _tqdm
    except ImportError:
        return None


def progress(iterable: Iterable[_T], /, **kwargs: Any) -> Iterator[_T]:
    """Wrap ``iterable`` in a tqdm bar if available, else iterate transparently.

    Extra keyword arguments (``desc``, ``total``, ``leave`` ...) are forwarded to
    ``tqdm`` when it is installed and ignored otherwise.
    """
    tqdm = _tqdm_or_none()
    if tqdm is None:
        return iter(iterable)
    return iter(tqdm.tqdm(iterable, **kwargs))


def trange(*args: int, **kwargs: Any) -> Iterator[int]:
    """``tqdm.trange`` if available, else plain :func:`range`."""
    tqdm = _tqdm_or_none()
    if tqdm is None:
        return iter(range(*args))
    return iter(tqdm.trange(*args, **kwargs))
