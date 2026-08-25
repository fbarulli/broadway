FROM python:3.12-slim AS build

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

FROM python:3.12-slim AS runtime

# lightgbm dlopens libgomp.so.1 at import (training stack); the slim base
# lacks it — same install the k8s/optuna/Dockerfile.base runtime carries.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /app/.venv /app/.venv
COPY src/ /app/src/
COPY scripts/ /app/scripts/
COPY configs/ /app/configs/

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH=/app/src

CMD ["ds-pipeline"]
