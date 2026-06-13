#!/usr/bin/env bash
set -euo pipefail

DATASETS="${DATASETS:-medical hotpotqa musique 2wikimultihop novel}"
BACKENDS="${BACKENDS:-faiss-flat faiss-pq turbovec}"
LIMIT="${LIMIT:-5}"
TOP_K="${TOP_K:-4}"
MAX_ROUNDS="${MAX_ROUNDS:-3}"
CONFIG="${CONFIG:-config.yaml}"
EMBEDDING_PROVIDER="${EMBEDDING_PROVIDER:-}"
EMBEDDING_MODEL_PATH="${EMBEDDING_MODEL_PATH:-}"
EMBEDDING_BATCH_SIZE="${EMBEDDING_BATCH_SIZE:-}"
EMBEDDING_ONLY="${EMBEDDING_ONLY:-}"
RAG_ONLY="${RAG_ONLY:-}"
RESULTS_DIR="${RESULTS_DIR:-results}"
RUN_NAME="${RUN_NAME:-faiss_vs_turbovec_llm_limit${LIMIT}}"
OUTPUT_JSON="${OUTPUT_JSON:-${RESULTS_DIR}/${RUN_NAME}.json}"
OUTPUT_LOG="${OUTPUT_LOG:-${RESULTS_DIR}/${RUN_NAME}.log}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
BENCHMARK_BIN="${BENCHMARK_BIN:-.venv/bin/agentic-rag-benchmark}"
UV_BIN="${UV_BIN:-uv}"
UV_CACHE_DIR="${UV_CACHE_DIR:-.cache/uv}"
PYPI_INDEX_URL="${PYPI_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"

mkdir -p "${RESULTS_DIR}"
mkdir -p "${UV_CACHE_DIR}"
export UV_CACHE_DIR

if ! command -v "${UV_BIN}" >/dev/null 2>&1; then
  echo "uv is required but was not found on PATH." >&2
  exit 1
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Project virtualenv not found at ${PYTHON_BIN}; creating it with uv." >&2
  "${UV_BIN}" venv
fi

if [[ ! -x "${BENCHMARK_BIN}" ]]; then
  echo "Installing this project into the virtualenv..."
  "${UV_BIN}" pip install --python "${PYTHON_BIN}" -e .
fi

if ! "${PYTHON_BIN}" -c "import faiss, numpy, sentence_transformers, turbovec" >/dev/null 2>&1; then
  echo "Installing benchmark vector backends into ${PYTHON_BIN}..."
  "${UV_BIN}" pip install \
    --python "${PYTHON_BIN}" \
    --index-url "${PYPI_INDEX_URL}" \
    numpy faiss-cpu sentence-transformers turbovec
fi

echo "Starting AgenticRAG FAISS/turbovec experiment"
echo "  config:   ${CONFIG}"
echo "  datasets: ${DATASETS}"
echo "  backends: ${BACKENDS}"
echo "  limit:    ${LIMIT}"
if [[ -n "${EMBEDDING_MODEL_PATH}" ]]; then
  echo "  embedding override: ${EMBEDDING_MODEL_PATH}"
fi
echo "  log:      ${OUTPUT_LOG}"
echo "  json:     ${OUTPUT_JSON}"
echo

export PYTHONUNBUFFERED=1
EXTRA_ARGS=()
if [[ -n "${EMBEDDING_PROVIDER}" ]]; then
  EXTRA_ARGS+=(--embedding-provider "${EMBEDDING_PROVIDER}")
fi
if [[ -n "${EMBEDDING_MODEL_PATH}" ]]; then
  EXTRA_ARGS+=(--embedding-model-path "${EMBEDDING_MODEL_PATH}")
fi
if [[ -n "${EMBEDDING_BATCH_SIZE}" ]]; then
  EXTRA_ARGS+=(--embedding-batch-size "${EMBEDDING_BATCH_SIZE}")
fi
if [[ -n "${EMBEDDING_ONLY}" ]]; then
  EXTRA_ARGS+=(--embedding-only)
fi
if [[ -n "${RAG_ONLY}" ]]; then
  EXTRA_ARGS+=(--rag-only)
fi
CMD=(
  "${BENCHMARK_BIN}"
  --config "${CONFIG}" \
  --datasets ${DATASETS} \
  --limit "${LIMIT}" \
  --backends ${BACKENDS} \
  --top-k "${TOP_K}" \
  --max-rounds "${MAX_ROUNDS}" \
  --output "${OUTPUT_JSON}"
)
if ((${#EXTRA_ARGS[@]})); then
  CMD+=("${EXTRA_ARGS[@]}")
fi

set -x
"${CMD[@]}" 2>&1 | tee "${OUTPUT_LOG}"
