from __future__ import annotations

import contextlib
import functools
import importlib
import logging
from collections.abc import Callable
from enum import Enum
from typing import Any, ParamSpec, TypeVar, cast

from panda_matching.config import get_settings

P = ParamSpec("P")
R = TypeVar("R")


class SpanType(str, Enum):
    AGENT = "AGENT"
    CHAT_MODEL = "CHAT_MODEL"
    TOOL = "TOOL"


_TRACING_CONFIGURED = False
logger = logging.getLogger(__name__)


def _get_mlflow() -> Any | None:
    try:
        return importlib.import_module("mlflow")
    except ImportError:  # pragma: no cover - optional dependency
        return None


def tracing_enabled() -> bool:
    settings = get_settings()
    return (
        _get_mlflow() is not None
        and bool(settings.mlflow_tracking_uri)
        and bool(settings.mlflow_experiment_id)
    )


def _mlflow_span_type(span_type: SpanType | None) -> Any:
    mlflow = _get_mlflow()
    if span_type is None or mlflow is None:
        return None
    mlflow_span_type = getattr(mlflow.entities.span, "SpanType", None)
    if mlflow_span_type is None:
        return None
    return getattr(mlflow_span_type, span_type.name, None)


def configure_mlflow_tracing() -> None:
    global _TRACING_CONFIGURED
    if _TRACING_CONFIGURED or not tracing_enabled():
        return

    settings = get_settings()
    mlflow = _get_mlflow()
    assert mlflow is not None
    assert settings.mlflow_tracking_uri is not None
    assert settings.mlflow_experiment_id is not None
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(experiment_id=settings.mlflow_experiment_id)
    _TRACING_CONFIGURED = True


def trace(
    *,
    name: str | None = None,
    span_type: SpanType | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        if _get_mlflow() is None:
            return func

        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            if not tracing_enabled():
                return func(*args, **kwargs)
            configure_mlflow_tracing()
            mlflow = _get_mlflow()
            assert mlflow is not None
            traced = mlflow.trace(
                name=name or func.__name__,
                span_type=_mlflow_span_type(span_type),
            )(func)
            try:
                return cast(R, traced(*args, **kwargs))
            except RuntimeError as exc:
                if "generator didn't stop after throw()" not in str(exc):
                    raise
                logger.exception(
                    "MLflow tracing wrapper failed for %s; retrying without tracing",
                    name or func.__name__,
                )
                return func(*args, **kwargs)

        return wrapper

    return decorator


def start_span(
    name: str,
    *,
    span_type: SpanType | None = None,
    attributes: dict[str, Any] | None = None,
) -> Any:
    mlflow = _get_mlflow()
    if mlflow is None or not tracing_enabled():
        return contextlib.nullcontext(None)

    configure_mlflow_tracing()
    return mlflow.start_span(
        name=name,
        span_type=_mlflow_span_type(span_type),
        attributes=attributes,
    )
