FROM ghcr.io/astral-sh/uv:0.12.23 AS uv
FROM python:3.12.15-slim

COPY --from=uv /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_PYTHON_DOWNLOADS=never
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --locked --no-dev --no-editable

ENTRYPOINT ["/app/.venv/bin/bank-marketing-ml"]
