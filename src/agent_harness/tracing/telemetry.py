from __future__ import annotations

import contextlib
import functools
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import SERVICE_NAME, Resource


@dataclass
class TelemetryConfig:
    service_name: str = "agent-harness"
    otlp_endpoint: str = "http://localhost:4317"
    enabled: bool = False
    sample_rate: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class TraceManager:
    def __init__(self, config: Optional[TelemetryConfig] = None):
        self._config = config or TelemetryConfig()
        self._tracer: Optional[trace.Tracer] = None
        if self._config.enabled:
            self._setup()

    def _setup(self):
        resource = Resource(attributes={SERVICE_NAME: self._config.service_name})
        provider = TracerProvider(resource=resource)
        if self._config.otlp_endpoint:
            exporter = OTLPSpanExporter(endpoint=self._config.otlp_endpoint, insecure=True)
            provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        self._tracer = trace.get_tracer(self._config.service_name)

    @property
    def tracer(self) -> trace.Tracer:
        if self._tracer is None:
            self._tracer = trace.get_tracer(self._config.service_name)
        return self._tracer

    def start_span(self, name: str, attributes: Dict[str, Any] = None) -> trace.Span:
        return self.tracer.start_span(name, attributes=attributes or {})

    @contextlib.contextmanager
    def span(self, name: str, **attrs):
        with self.tracer.start_as_current_span(name, attributes=attrs) as span:
            yield span


def traced(name: str = None, attrs: Dict[str, Any] = None):
    def decorator(func: Callable):
        span_name = name or func.__name__
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            tracer = trace.get_tracer("agent-harness")
            with tracer.start_as_current_span(span_name, attributes=attrs or {}) as span:
                start = time.monotonic()
                try:
                    result = await func(*args, **kwargs)
                    span.set_attribute("duration_ms", (time.monotonic() - start) * 1000)
                    return result
                except Exception as e:
                    span.set_attribute("error", str(e))
                    span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                    raise
        return wrapper
    return decorator
