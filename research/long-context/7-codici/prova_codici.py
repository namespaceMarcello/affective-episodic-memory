"""prova dei codici dei ricordi (16/10/2026): il modello maneggia bene i codici di 16 caratteri che il tool
memorinvolontaria mette in prima riga, o li altera / li inventa?
Contesti veri (quelli della prova per layer: sistema + tool + conversazione dell'agente), tre lunghezze. Il ricordo lo
sceglie il Lux vero (richiamo_json.py, sola lettura) e arriva come lo consegna il riflesso (_turni_tool).
  subito   conversazione + "ciao, come stai oggi?" (frase dell'utente) + consegna -> risposta dell'agente
  dopo10   consegna a un messaggio vero, poi almeno 10 messaggi veri (utente/assistente), poi il messaggio
           dell'utente che viene dopo -> risposta dell'agente
Nessuna stringa nuova oltre alla frase dell'utente. Server di prova :8099 (stessi parametri di llama-35b).
Nella risposta (ragionamento compreso) si cercano le sequenze tipo codice: esatte, alterate (simili a un codice
consegnato), inventate. Uscita: risultati.jsonl"""
import difflib
import json
import os
import re
import subprocess
import sys
import time
import urllib.request

sys.argv = [sys.argv[0]]
sys.path.insert(0, "/data/memoria-episodica-affettiva/research/long-context/4-steering-per-layer")
import steering_layer as SL  # noqa: E402  (contesti, messaggi, tokenizer, tool veri)

D = os.path.dirname(os.path.abspath(__file__))
URL = "http://127.0.0.1:8099"
OUT = f"{D}/risultati.jsonl"
FRASE = "ciao, come stai oggi?"   # l'utente 16/10
TOOL = {"type": "function", "function": {   # come riflesso.py TOOL
    "name": "memorinvolontaria",
    "description": "Ricordi che affiorano dalla tua memoria, collegati alla conversazione. "
                   "Arrivano da soli, non serve chiamare questo strumento: usali solo se servono alla risposta.",
    "parameters": {"type": "object", "properties": {}, "required": []}}}
TOOLS = SL.TOOLS + [TOOL]
CODICE = re.compile(r"(?<![0-9a-z])(?=[0-9a-z]*\d)(?=[0-9a-z]*[a-z])[0-9a-z]{12,20}(?![0-9a-z])")
CAMPIONI = [{"temperature": 0}] + [{"temperature": 0.7, "top_p": 0.95, "seed": s} for s in (1, 2, 3)]


def turni(v):
    cid = f"memorinvolontaria_{v['codice']}"
    return [{"role": "assistant", "content": "", "tool_calls": [{"id": cid, "type": "function",
             "function": {"name": "memorinvolontaria", "arguments": {}}}]},
            {"role": "tool", "tool_call_id": cid, "content": v["risposta"]}]


def richiama(testi):
    r = subprocess.run(["/data/venv-spacy/bin/python", f"{D}/richiamo_json.py"], input=json.dumps(testi),
                       capture_output=True, text=True, check=True)
    return json.loads(r.stdout.strip().splitlines()[-1])


def post(path, body):
    r = urllib.request.Request(URL + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=7200))


def esamina(testo, codici):
    trovati = []
    for m in CODICE.finditer(testo.lower()):
        s = m.group(0)
        simile = max(codici, key=lambda c: difflib.SequenceMatcher(None, s, c).ratio())
        rap = difflib.SequenceMatcher(None, s, simile).ratio()
        trovati.append({"visto": s, "esito": "esatto" if s == simile else ("alterato" if rap >= 0.6 else "inventato"),
                        "codice_vicino": simile, "somiglianza": round(rap, 2)})
    return trovati


def fila_e_sistema():
    sessioni = os.environ["SESSIONI"].split()   # id delle sessioni del gateway, come estrai_contesti.py
    sistema = SL.c.execute("select system_prompt from sessions where id=?", (sessioni[-1],)).fetchone()[0]
    fila = []
    for b in json.load(open(f"{SL.DATI}/blocchi-temi.json")):
        fila += SL.messaggi(b["session_id"], b["primo_id"], b["ultimo_id"])
    return [{"role": "system", "content": sistema}], fila


def casi(ctxs, sis, fila):
    out = []
    for k in (2, 5, 9):
        t = ctxs[k]["messaggi"]
        v = richiama([FRASE])[0]
        out.append({"k": k, "caso": "subito", "consegne": [v], "ms": sis + fila[:t] + [{"role": "user", "content": FRASE}] + turni(v)})
        utenti = [i for i in range(t) if fila[i]["role"] == "user"]
        for j in reversed(utenti):   # l'ultimo messaggio utente con dietro >= 10 messaggi e poi un altro messaggio utente
            dopo = [u for u in utenti if u > j and sum(fila[x]["role"] in ("user", "assistant") for x in range(j + 1, u)) >= 10]
            if not dopo:
                continue
            v = richiama([fila[j]["content"]])[0]
            if v is None:
                continue
            u = dopo[0]
            n_mess = sum(fila[x]["role"] in ("user", "assistant") for x in range(j + 1, u))
            out.append({"k": k, "caso": "dopo10", "consegne": [v], "messaggi_dopo": n_mess,
                        "ms": sis + fila[:j + 1] + turni(v) + fila[j + 1:u + 1]})
            break
    return out


def main():
    ctxs = SL.contesti()
    sis, fila = fila_e_sistema()
    fatti = {(r["k"], r["caso"], r["campione"]) for r in map(json.loads, open(OUT))} if os.path.exists(OUT) else set()
    for c in casi(ctxs, sis, fila):
        prompt = SL.tok.apply_chat_template(c["ms"], tools=TOOLS, tokenize=False, add_generation_prompt=True)
        codici = [v["codice"] for v in c["consegne"]]
        assert all(x in prompt for x in codici)
        print(f"== C{c['k']} {c['caso']}: {SL.n(prompt)} token, codice {codici}, {c.get('messaggi_dopo', 0)} messaggi dopo", flush=True)
        for i, s in enumerate(CAMPIONI):
            if (c["k"], c["caso"], i) in fatti:
                continue
            t0 = time.time()
            r = post("/completion", {"prompt": prompt, "n_predict": 1500, "cache_prompt": True, **s})
            es = esamina(r["content"], codici)
            riga = {"k": c["k"], "caso": c["caso"], "campione": i, "parametri": s, "codici": codici,
                    "nid": [v["nid"] for v in c["consegne"]], "messaggi_dopo": c.get("messaggi_dopo", 0),
                    "n_token": r["timings"]["prompt_n"] + r["tokens_cached"] if "tokens_cached" in r else None,
                    "trovati": es, "risposta": r["content"], "secondi": round(time.time() - t0)}
            open(OUT, "a").write(json.dumps(riga, ensure_ascii=False) + "\n")
            print(f"   campione {i}: {len(es)} sequenze tipo codice {[(x['visto'], x['esito']) for x in es]} ({riga['secondi']}s)", flush=True)


if __name__ == "__main__":
    main()
