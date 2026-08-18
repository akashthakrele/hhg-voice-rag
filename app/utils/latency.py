"""
Utility decorators and helpers for latency instrumentation.
Every pipeline stage is timed and logged.
"""

from __future__ import annotations

import functools
import time
from collections.abc import Callable
from typing import Any

import structlog

from app.core.config import get_settings

logger = structlog.get_logger(__name__)


def latency_timer(stage_name: str) -> Callable:
    """
    Decorator that measures execution time of a function in milliseconds.

    Usage:
        @latency_timer("retrieval")
        async def retrieve(query: str) -> list:
            ...

    The decorated function receives an injected `_timings` dict (if passed)
    where `stage_name` → elapsed_ms is recorded.
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            settings = get_settings()
            start = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                elapsed_ms = (time.perf_counter() - start) * 1000
                exceeds = elapsed_ms > settings.latency_target_ms

                logger.info(
                    "stage_latency",
                    stage=stage_name,
                    elapsed_ms=round(elapsed_ms, 2),
                    exceeds_target=exceeds,
                    target_ms=settings.latency_target_ms,
                )

                # Inject timing into the _timings dict if provided
                timings = kwargs.get("_timings")
                if timings is not None:
                    timings[stage_name] = round(elapsed_ms, 2)

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            settings = get_settings()
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                elapsed_ms = (time.perf_counter() - start) * 1000
                exceeds = elapsed_ms > settings.latency_target_ms

                logger.info(
                    "stage_latency",
                    stage=stage_name,
                    elapsed_ms=round(elapsed_ms, 2),
                    exceeds_target=exceeds,
                    target_ms=settings.latency_target_ms,
                )

                timings = kwargs.get("_timings")
                if timings is not None:
                    timings[stage_name] = round(elapsed_ms, 2)

        import asyncio

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator
