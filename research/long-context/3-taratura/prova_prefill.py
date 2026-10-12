#!/usr/bin/env python3
"""Prova del prefill colorato del riflesso (12/10/2026, design dell'autrice del 10/10): il vettore
emotivo acceso SOLO mentre passa il testo del ricordo, poi spento, e la risposta riusa la cache.
Usa le funzioni vere di riflesso.py (_rimetti, _colora_ricordo) contro il server di prova :8099, con una
conversazione vera dell'agente (state.db, copia del database del gateway non inclusa, come estrai_contesti.py, finestra ~32k token) e un ricordo vero
(memoria.db) nel turno del tool memorinvolontaria. Nessuna stringa nuova.
Misure: token rielaborati a ogni passo (timings.prompt_n del server): dopo il prefill colorato la risposta deve
rielaborare solo la coda (pochi token), non la conversazione. Poi la stessa risposta col ricordo NON colorato,
per vedere se il colore cambia qualcosa. Uscita: risultati-prefill.json e righe su stdout."""
import json
import os
import sqlite3
import sys

os.environ["PONTE_URL"] = "http://127.0.0.1:8099"
sys.path.insert(0, "/data/memoria-episodica-affettiva")
import ponte  # noqa: E402
import riflesso  # noqa: E402

import transformers  # noqa: E402

D = "/data/memoria-episodica-affettiva/research/long-context/3-taratura"
DATI = "/data/memoria-episodica-affettiva/research/long-context/dati"
SESSIONI = os.environ["SESSIONI"].split()   # id delle sessioni del gateway, come estrai_contesti.py
BUDGET = 32768
RICORDO = int(os.environ["RICORDO"])   # id del ricordo da mettere nel turno del tool

passi = []
_post = ponte.post


def post_misurato(path, payload, timeout=600):
    st, r = _post(path, payload, timeout)
    t = r.get("timings", {}) if isinstance(r, dict) else {}
    passi.append({"path": path, "stato": st, "prompt_n": t.get("prompt_n"), "prompt_ms": t.get("prompt_ms")})
    print(f"   {path}: stato {st}, token rielaborati {t.get('prompt_n')}, {t.get('prompt_ms', 0) / 1000:.1f}s", flush=True)
    return st, r


ponte.post = post_misurato
riflesso._diario = lambda voce: print("   diario:", voce, flush=True)

# --- conversazione vera, finestra ~32k
tok = transformers.AutoTokenizer.from_pretrained("Accio-Lab/occamy-1.0")
c = sqlite3.connect(f"file:{DATI}/state.db?mode=ro", uri=True)
sistema = c.execute("select system_prompt from sessions where id=?", (SESSIONI[-1],)).fetchone()[0]
msgs = []
for sid in SESSIONI:
    for role, content, tool_calls, call_id in c.execute(
            "select role, content, tool_calls, tool_call_id from messages where session_id=? order by id", (sid,)):
        if role == "assistant":
            m = {"role": "assistant", "content": content or ""}
            if tool_calls:
                m["tool_calls"] = [{"id": t.get("id", f"c{i}"), "type": "function",
                                    "function": {"name": t["function"]["name"],
                                                 "arguments": t["function"]["arguments"] or "{}"}}
                                   for i, t in enumerate(json.loads(tool_calls))]
            msgs.append(m)
        elif role == "tool":
            msgs.append({"role": "tool", "content": content or "", "tool_call_id": call_id})
        elif role == "user" and (content or "").strip():
            msgs.append({"role": "user", "content": content})
fine = max(i for i, m in enumerate(msgs) if m["role"] == "user" and len(m["content"]) > 40)
msgs = msgs[:fine + 1]
inizio = len(msgs) - 1
def per_template(ms):   # il template HF vuole gli arguments come dict, il server come stringa
    return [dict(m, tool_calls=[dict(t, function=dict(t["function"], arguments=json.loads(t["function"]["arguments"])))
                                for t in m["tool_calls"]]) if "tool_calls" in m else m for m in ms]


while inizio > 0 and len(tok.apply_chat_template(per_template([{"role": "system", "content": sistema}] + msgs[inizio - 1:]),
                                                  tokenize=False)) < BUDGET * 3:   # ~3 caratteri a token
    inizio -= 1
while msgs[inizio]["role"] == "tool":
    inizio += 1
conv = [{"role": "system", "content": sistema}] + msgs[inizio:]

# --- ricordo vero, nel formato della consegna
m = sqlite3.connect("file:/data/workspace/memoria/memoria.db?mode=ro", uri=True)
testo, ts, emo = m.execute("select testo, ts, emo_tag from nodi where id=?", (RICORDO,)).fetchone()
codice = riflesso._codice(ts)
voce = {"dopo": riflesso._impronta(conv[-1]["content"]), "codice": codice, "risposta": codice + "\n" + testo}


def richiesta(con_ricordo):
    d = {"messages": json.loads(json.dumps(conv)), "tools": [riflesso.TOOL]}
    if con_ricordo:
        riflesso._rimetti(d["messages"], [voce])
    return d


def chat(d, n):
    st, r = ponte.post("/v1/chat/completions", dict(d, max_tokens=n, temperature=0, stream=False), timeout=3600)
    msg = r["choices"][0]["message"] if st == 200 else {}
    return (msg.get("reasoning_content") or "") + (msg.get("content") or "")


ris = {"ricordo": RICORDO, "emo": emo, "n_messaggi": len(conv)}
print(f"== conversazione: {len(conv)} messaggi; ricordo {RICORDO} ({emo})", flush=True)
ponte.calma()
print("== 1. la conversazione entra nella cache (come la chat dell'agente prima del messaggio)", flush=True)
chat(richiesta(False), 1)
print("== 2. prefill colorato del ricordo (riflesso._colora_ricordo)", flush=True)
d = richiesta(True)
ris["colorato"] = riflesso._colora_ricordo(d, voce["risposta"], emo)
print("== 3. la risposta: deve rielaborare solo la coda", flush=True)
ris["testo_colorato"] = chat(d, 96)
print(f"   {ris['testo_colorato'][:300]!r}", flush=True)
print("== 4. stesso ricordo NON colorato (cache riportata a prima del ricordo)", flush=True)
p = ponte.post("/apply-template", {k: v for k, v in d.items() if k in ("messages", "tools")})[1]["prompt"]
a = p.rfind("<tool_response>", 0, p.rfind(voce["risposta"])) + len("<tool_response>")
ponte.post("/completion", {"prompt": p[:a], "n_predict": 1, "cache_prompt": True})
ris["testo_neutro"] = chat(d, 96)
print(f"   {ris['testo_neutro'][:300]!r}", flush=True)
ris["passi"] = passi
ris["uguali"] = ris["testo_colorato"] == ris["testo_neutro"]
print(f"== colorato e neutro {'UGUALI' if ris['uguali'] else 'DIVERSI'}", flush=True)
json.dump(ris, open(f"{D}/risultati-prefill.json", "w"), indent=1, ensure_ascii=False)
