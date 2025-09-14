#!/usr/bin/env bash
set -e
echo "Pulling free Ollama models..."
ollama pull llama3:instruct
ollama pull llava || true
echo "Done."