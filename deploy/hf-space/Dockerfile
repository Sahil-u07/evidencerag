# Layout (deploy.py builds it): Dockerfile, start.sh, README.md and LICENSE at the Space root,
# next to backend/, frontend/ and data/evaluation/v2/corpus/. The build context is that root.

FROM node:22-slim AS web
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
# Empty means the page calls the API on its own origin.
ENV VITE_API_BASE=
RUN npm run build


FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl zstd \
    && rm -rf /var/lib/apt/lists/*

# The CUDA, ROCm and MLX libraries are left out: the Space has no GPU.
ARG OLLAMA_VERSION=v0.40.15
ARG OLLAMA_SHA256=628c018404a4206493efd38e46251c154ede5141b8b8dc85c659b3b027c47379
RUN curl -fsSL -o /tmp/ollama.tar.zst \
        "https://github.com/ollama/ollama/releases/download/${OLLAMA_VERSION}/ollama-linux-amd64.tar.zst" \
    && echo "${OLLAMA_SHA256}  /tmp/ollama.tar.zst" | sha256sum -c - \
    && tar --zstd -xf /tmp/ollama.tar.zst -C /usr/local \
        --exclude='lib/ollama/cuda_*' --exclude='lib/ollama/rocm*' --exclude='lib/ollama/mlx*' \
    && rm /tmp/ollama.tar.zst

# Spaces run the container as uid 1000.
RUN useradd -m -u 1000 user \
    && mkdir -p /app/data/uploads /app/data/index /app/licenses \
    && chown -R user:user /app

ENV HOME=/home/user \
    HF_HOME=/home/user/.cache/huggingface \
    OLLAMA_MODELS=/home/user/.ollama/models \
    OLLAMA_HOST=127.0.0.1:11434 \
    OLLAMA_NUM_PARALLEL=1 \
    OLLAMA_MAX_LOADED_MODELS=1 \
    OLLAMA_KEEP_ALIVE=30m \
    PYTHONUNBUFFERED=1 \
    READ_ONLY=true \
    FRONTEND_DIST=/app/frontend/dist \
    UPLOAD_DIR=/app/data/uploads \
    INDEX_STORAGE_DIR=/app/data/index \
    OLLAMA_URL=http://127.0.0.1:11434

USER user
WORKDIR /app/backend

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY --chown=user:user backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Pull the model while a temporary server runs, and keep its licence text.
RUN ollama serve >/tmp/ollama-build.log 2>&1 & pid=$!; \
    until curl -fs http://127.0.0.1:11434/api/tags >/dev/null; do sleep 1; done; \
    ollama pull qwen2.5:3b \
    && ollama show qwen2.5:3b --license > /app/licenses/qwen2.5-3b-LICENSE.txt \
    && test -s /app/licenses/qwen2.5-3b-LICENSE.txt \
    && kill $pid && wait $pid

RUN python -c "from sentence_transformers import CrossEncoder, SentenceTransformer; \
SentenceTransformer('all-MiniLM-L6-v2'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2'); \
CrossEncoder('cross-encoder/nli-MiniLM2-L6-H768')"

COPY --chown=user:user backend/ ./
COPY --chown=user:user data/evaluation/v2/corpus/rfc* /app/data/uploads/
# Build the index now so the first request does not wait for it.
RUN python -c "from app import indexing; indexing.build_retriever()"

COPY --from=web --chown=user:user /web/dist /app/frontend/dist
COPY --chown=user:user LICENSE /app/LICENSE
COPY --chown=user:user start.sh /app/start.sh

# pip installs as this user under ~/.local, so uvicorn lives there.
ENV PATH=/home/user/.local/bin:$PATH
# Models are all in the image, so nothing is fetched at run time.
ENV HF_HUB_OFFLINE=1
EXPOSE 7860
CMD ["sh", "/app/start.sh"]
