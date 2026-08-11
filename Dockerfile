FROM python:3.12-slim AS build

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

FROM python:3.12-slim AS runtime

COPY --from=build /app/.venv /app/.venv
COPY src/ /app/src/
COPY scripts/ /app/scripts/
COPY configs/ /app/configs/

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH=/app/src

CMD ["ds-pipeline"]
