#!/bin/sh
set -e

ollama serve >/tmp/ollama.log 2>&1 &

tries=0
until curl -fs http://127.0.0.1:11434/api/tags >/dev/null; do
    tries=$((tries + 1))
    if [ "$tries" -ge 60 ]; then
        echo "Ollama did not start within 60 seconds." >&2
        cat /tmp/ollama.log >&2
        exit 1
    fi
    sleep 1
done

# Load the model now, with the same context size the app asks for, so the first visitor does not wait.
curl -s http://127.0.0.1:11434/api/generate \
    -d '{"model":"qwen2.5:3b","prompt":"hi","stream":false,"keep_alive":"30m","options":{"num_ctx":2048,"num_predict":1}}' \
    >/dev/null 2>&1 &

exec uvicorn app.main:app --host 0.0.0.0 --port 7860
