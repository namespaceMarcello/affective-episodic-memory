#!/bin/bash
# 14/10: verifica "fine ragionamento al primo punto + grammatica della tool call" a prova ferma. Server come lancia.sh (stessi parametri), poi
# verifica_think.py; il server si spegne a fine lavoro.
D=/data/memoria-episodica-affettiva/research/long-context; cd $D
${LUXIFER_PROVA:-/data/luxifer-prova}/build-prova/bin/llama-server -m /data/gguf/occamy-1.0/occamy-1.0-Q8_0.gguf \
  --port 8099 --host 127.0.0.1 --n-gpu-layers 99 --n-cpu-moe 34 --flash-attn on --threads 8 --threads-batch 8 -np 1 \
  -c 143360 --repeat-penalty 1.1 --slot-save-path $D/4-steering-per-layer/slot/ > server-verifica-grammatica.log 2>&1 &
SPID=$!
trap 'kill $SPID 2>/dev/null' EXIT
for i in $(seq 1 300); do curl -sf -m 2 http://127.0.0.1:8099/health >/dev/null && break; kill -0 $SPID 2>/dev/null || { echo "server non partito"; exit 1; }; sleep 2; done
HF_HOME=/data/jspace/hf HF_HUB_OFFLINE=1 /data/jspace/venv/bin/python -u 4-steering-per-layer/verifica_grammatica.py 2>&1 | tee -a 4-steering-per-layer/log-verifica-grammatica.txt
