# Multi-stage: dependencies are resolved and installed in a builder, and only the
# resulting virtual environment and the project files reach the runtime image.

FROM python:3.12-slim AS builder
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
RUN python -m venv /venv
ENV PATH="/venv/bin:$PATH"
COPY pyproject.toml ./
COPY pipelines ./pipelines
COPY service ./service
RUN pip install ".[prod]"

FROM python:3.12-slim AS runtime
ENV PATH="/venv/bin:$PATH" PYTHONUNBUFFERED=1 HARDPOINT_ENV=prod
RUN useradd --create-home --uid 10001 app
WORKDIR /app
COPY --from=builder /venv /venv
COPY --chown=app:app . .
USER app
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz')" || exit 1
# Graceful shutdown: in-flight requests get the request deadline to finish.
CMD ["hardpoint", "serve", "--host", "0.0.0.0", "--port", "8000"]
