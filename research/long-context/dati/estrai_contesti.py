#!/usr/bin/env python3
"""Prova contesto lungo (11/10/2026): contesti VERI dell'agente, renderizzati col template di Occamy.
Fonte: state.db (copia in sola lettura del database del gateway agentico, non inclusa nel repo): sessioni
Matrix, prompt di sistema vero, turni utente/assistente/tool in ordine (anche quelli poi compattati: il contesto
vivo prima della compattazione). Il punto di prova e' l'ultimo messaggio utente con testo: il contesto finisce
li', pronto per la risposta dell'agente. Nessuna stringa nuova: solo testi gia' visti dall'agente.
Uscita: contesti.json = [{nome, n_token, testo}] con
  isolato   gli ultimi ~1k token di conversazione, senza prompt di sistema
  sys       prompt di sistema + gli ultimi ~1k token
  32k 64k 100k 140k   prompt di sistema + quanta conversazione ci sta
Uso: HF_HOME=/data/jspace/hf HF_HUB_OFFLINE=1 /data/jspace/venv/bin/python estrai_contesti.py <sessione> [<sessione> ...]
     (id delle sessioni in ordine di tempo: l'ultima finisce la prova)"""
import json
import sqlite3
import sys

import transformers

DB = "/data/memoria-episodica-affettiva/research/long-context/dati/state.db"
OUT = "/data/memoria-episodica-affettiva/research/long-context/dati/contesti.json"
SESSIONI = sys.argv[1:]   # in ordine di tempo: l'ultima finisce la prova
MISURE = {"32k": 32768, "64k": 65536, "100k": 100000, "140k": 140000}
CODA = 1024

tok = transformers.AutoTokenizer.from_pretrained("Accio-Lab/occamy-1.0")
c = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
sistema = c.execute("select system_prompt from sessions where id=?", (SESSIONI[-1],)).fetchone()[0]
msgs = []
for sid in SESSIONI:
    for role, content, tool_calls, call_id in c.execute(
            "select role, content, tool_calls, tool_call_id from messages where session_id=? order by id", (sid,)):
        if role == "assistant":
            m = {"role": "assistant", "content": content or ""}
            if tool_calls:
                m["tool_calls"] = [{"type": "function", "function": {"name": t["function"]["name"],
                                    "arguments": json.loads(t["function"]["arguments"] or "{}")}}
                                   for t in json.loads(tool_calls)]
            msgs.append(m)
        elif role == "tool":
            msgs.append({"role": "tool", "content": content or "", "tool_call_id": call_id})
        elif role == "user" and (content or "").strip():
            msgs.append({"role": "user", "content": content})
fine = max(i for i, m in enumerate(msgs) if m["role"] == "user" and len(m["content"]) > 40)
msgs = msgs[:fine + 1]


def rendi(con_sistema, inizio):
    m = ([{"role": "system", "content": sistema}] if con_sistema else []) + msgs[inizio:]
    return tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)


def n(testo):
    return len(tok(testo, add_special_tokens=False).input_ids)


def inizio_per(con_sistema, budget):
    """il primo messaggio da cui il contesto sta nel budget (taglio su confini di messaggio; mai su un tool orfano)"""
    lo, hi = 0, len(msgs) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if n(rendi(con_sistema, mid)) <= budget:
            hi = mid
        else:
            lo = mid + 1
    while msgs[lo]["role"] == "tool":
        lo += 1
    return lo


n_sis = n(rendi(True, len(msgs) - 1)) - n(rendi(False, len(msgs) - 1))
out = [{"nome": "isolato", "testo": rendi(False, inizio_per(False, CODA))},
       {"nome": "sys", "testo": rendi(True, inizio_per(True, n_sis + CODA))}]
for nome, budget in MISURE.items():
    out.append({"nome": nome, "testo": rendi(True, inizio_per(True, budget))})
for o in out:
    o["n_token"] = n(o["testo"])
    print(o["nome"], o["n_token"])
print(f"prompt di sistema ~{n_sis} token; conversazione totale {n(rendi(False, 0))} token; ultimo messaggio: {msgs[-1]['content'][:80]!r}")
json.dump(out, open(OUT, "w"), ensure_ascii=False)
