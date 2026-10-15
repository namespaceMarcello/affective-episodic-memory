#!/bin/bash
# 13/10: vettori sui passaggi in prima persona (aggiorna_emozioni.py). Seconda istanza di Occamy SOLO CPU sulla porta 8098
# (la GPU e il server 8099 restano alla prova per layer), si spegne da sola a fine lavoro.
cd /data/memoria-episodica-affettiva/research/long-context/6-template-lens
CUDA_VISIBLE_DEVICES= ${LUXIFER_PROVA:-/data/luxifer-prova}/build-prova/bin/llama-server \
  -m /data/gguf/occamy-1.0/occamy-1.0-Q8_0.gguf --port 8098 --host 127.0.0.1 -ngl 0 \
  --threads 4 --threads-batch 4 -np 1 -c 1024 > server-cpu-8098.log 2>&1 &
SPID=$!
trap 'kill $SPID 2>/dev/null' EXIT
for i in $(seq 1 300); do curl -sf -m 2 http://127.0.0.1:8098/health >/dev/null && break; kill -0 $SPID 2>/dev/null || { echo "server non partito: vedi server-cpu-8098.log"; exit 1; }; sleep 2; done
/data/jspace/venv/bin/python -u aggiorna_emozioni.py 2>&1 | tee -a log-prima-persona.txt
