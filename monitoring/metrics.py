"""Prometheus instrumentation ready for the dashboard component."""

from time import perf_counter

from fastapi import Response
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Histogram, generate_latest

BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10)


def instrument_api(app):
    registry = CollectorRegistry()
    requests = Counter("api_requests_total", "Completed API requests",
                       ("method", "path", "status"), registry=registry)
    duration = Histogram("api_request_duration_seconds", "API request duration",
                         ("method", "path"), buckets=BUCKETS, registry=registry)

    @app.middleware("http")
    async def observe(request, call_next):
        if request.url.path == "/metrics":
            return await call_next(request)
        started, status = perf_counter(), 500
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            route = request.scope.get("route")
            path = getattr(route, "path", "unmatched")
            requests.labels(request.method, path, str(status)).inc()
            duration.labels(request.method, path).observe(perf_counter() - started)

    @app.get("/metrics", include_in_schema=False)
    def metrics():
        return Response(generate_latest(registry), headers={"Content-Type": CONTENT_TYPE_LATEST})
