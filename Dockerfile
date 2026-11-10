FROM python:3.14-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.13 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project
COPY README.md ./
COPY src ./src
RUN uv sync --locked --no-dev

FROM python:3.14-slim
RUN useradd --create-home --uid 10001 meshguard
WORKDIR /app
COPY --from=build --chown=meshguard /app /app
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1
USER meshguard
RUN meshguard train
ENTRYPOINT ["meshguard"]
CMD ["run", "ddos", "--provider", "heuristic"]
