"""Phase 4a: Observability — Langfuse tracing and structured logging."""
from __future__ import annotations

import json
import logging
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Generator

logger = logging.getLogger("financial_advisor")


@dataclass
class TraceSpan:
    name: str
    start_time: float
    end_time: float = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    input_data: dict[str, Any] | None = None
    output_data: dict[str, Any] | None = None
    status: str = "OK"
    token_usage: dict[str, int] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000


class Tracer:
    """Lightweight tracing layer. Uses Langfuse if available, falls back to structured logging."""

    def __init__(self) -> None:
        self._langfuse = None
        self._trace = None
        self._spans: list[TraceSpan] = []
        self._total_tokens = {"prompt": 0, "completion": 0, "total": 0}
        self._init_langfuse()

    def _init_langfuse(self) -> None:
        public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
        secret_key = os.environ.get("LANGFUSE_SECRET_KEY")
        host = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")

        if public_key and secret_key:
            try:
                from langfuse import Langfuse
                self._langfuse = Langfuse(
                    public_key=public_key,
                    secret_key=secret_key,
                    host=host,
                )
                logger.info("Langfuse tracing enabled")
            except ImportError:
                logger.info("Langfuse not installed — using local tracing")
        else:
            logger.info("Langfuse keys not set — using local tracing")

    def start_trace(self, name: str, metadata: dict | None = None) -> None:
        if self._langfuse:
            self._trace = self._langfuse.trace(
                name=name,
                metadata=metadata or {},
            )
        logger.info(f"Trace started: {name}")

    @contextmanager
    def span(self, name: str, input_data: dict | None = None) -> Generator[TraceSpan, None, None]:
        s = TraceSpan(name=name, start_time=time.time(), input_data=input_data)
        langfuse_span = None

        if self._langfuse and self._trace:
            langfuse_span = self._trace.span(
                name=name,
                input=input_data,
            )

        try:
            yield s
        except Exception as e:
            s.status = f"ERROR: {e}"
            raise
        finally:
            s.end_time = time.time()
            self._spans.append(s)

            if langfuse_span:
                langfuse_span.end(
                    output=s.output_data,
                    metadata=s.metadata,
                    status_message=s.status,
                )

            logger.info(
                f"Span [{name}] completed in {s.duration_ms:.0f}ms | "
                f"status={s.status}"
            )

    def log_llm_call(self, name: str, model: str, prompt: str, response: str,
                     tokens: dict[str, int] | None = None) -> None:
        usage = tokens or {}
        self._total_tokens["prompt"] += usage.get("prompt", 0)
        self._total_tokens["completion"] += usage.get("completion", 0)
        self._total_tokens["total"] += usage.get("total", 0)

        if self._langfuse and self._trace:
            self._trace.generation(
                name=name,
                model=model,
                input=prompt,
                output=response,
                usage=usage,
            )

        logger.info(
            f"LLM call [{name}] model={model} | "
            f"tokens: {usage.get('total', 'N/A')}"
        )

    def get_summary(self) -> dict[str, Any]:
        total_duration = sum(s.duration_ms for s in self._spans)
        return {
            "total_spans": len(self._spans),
            "total_duration_ms": round(total_duration, 1),
            "token_usage": self._total_tokens.copy(),
            "spans": [
                {
                    "name": s.name,
                    "duration_ms": round(s.duration_ms, 1),
                    "status": s.status,
                }
                for s in self._spans
            ],
        }

    def flush(self) -> None:
        if self._langfuse:
            self._langfuse.flush()
        summary = self.get_summary()
        logger.info(f"Trace summary: {json.dumps(summary, indent=2)}")
