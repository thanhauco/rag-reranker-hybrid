# Offline-capable image (BM25 + hashing embedder + heuristic reranker + API).
# Build with models:  docker build --build-arg EXTRAS="api,models,llm" -t hybrid-rag .
FROM python:3.12-slim AS base
ARG EXTRAS="api"
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
RUN pip install ".[${EXTRAS}]"
RUN useradd --create-home app
USER app
EXPOSE 8000
CMD ["hybrid-rag", "serve", "--config", "configs/hybrid_heuristic.yaml", "--host", "0.0.0.0", "--port", "8000"]
