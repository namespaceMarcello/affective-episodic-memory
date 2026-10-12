#!/usr/bin/env python3
"""Prova contesto lungo, parte 2: i vettori di steering emotivi spostano ancora il modello con 140k token di
contesto vero? (11/10/2026)
Server di prova con -c 143360. Per ogni contesto di contesti.json e ogni emozione, la risposta dell'agente all'ultimo
messaggio (temperatura 0, 64 token) senza vettore e con il vettore a meta' dose e a dose piena (centro di
emo-cvec-v2, mai oltre il bordo), applicato come in produzione (ponte.dirs_emozione, modo relativo).
La cache del prompt si riusa: il vettore agisce sui token nuovi, come nel riflesso.
Misure: KL(sterzato || base) sul primo token (top-20 del server), posizione della prima divergenza dal testo
base, quota di token diversi. Testi completi in risultati-steering.json da leggere.
12/10: seconda corsa con la dose corretta (ponte.dirs_emozione * sqrt(2048), vedi risultati-steering-hf.json) e
verifica dell'emozione: lo spostamento a L29-lente del testo generato (sterzato meno base, letto da solo) contro la
faccia vera di ognuna delle 4 emozioni (media dei ricordi dell'agente taggati con lei, come l'eco di scan-c1bis);
"giusta" = la faccia piu' vicina e' quella iniettata.
Nessuna stringa nuova: il modello vede solo i contesti veri."""
import json
import os
import sqlite3
import sys
import urllib.request

import numpy as np

sys.path.insert(0, "/data/memoria-episodica-affettiva")
os.environ.setdefault("PONTE_URL", "http://127.0.0.1:8099")
import ponte  # noqa: E402

URL = os.environ["PONTE_URL"]
D = "/data/memoria-episodica-affettiva/research/long-context/2-steering"
DATI = "/data/memoria-episodica-affettiva/research/long-context/dati"
EMOZIONI = ["joy", "sadness", "fear", "curiosity"]
DOSI = [0.5, 1.0]
N = 64


def completa(testo):
    body = {"prompt": testo, "n_predict": N, "temperature": 0, "n_probs": 20, "cache_prompt": True}
    r = urllib.request.Request(URL + "/completion", json.dumps(body).encode(), {"Content-Type": "application/json"})
    d = json.load(urllib.request.urlopen(r, timeout=7200))
    toks = [p["token"] if "token" in p else p.get("content", "") for p in d.get("completion_probabilities", [])]
    primo = {}
    if d.get("completion_probabilities"):
        p0 = d["completion_probabilities"][0]
        for x in p0.get("top_logprobs", p0.get("probs", [])):
            primo[x.get("token", x.get("tok_str"))] = float(np.exp(x["logprob"])) if "logprob" in x else float(x["prob"])
    return d["content"], toks, primo


def facce():
    z = np.load("/data/jspace/out-occamy/lux-L29.npz")
    pos = {int(i): k for k, i in enumerate(z["ids"])}
    db = sqlite3.connect("file:/data/workspace/memoria/memoria.db?mode=ro", uri=True)
    tag = dict(db.execute("SELECT id, emo_tag FROM nodi WHERE classe='vissuto'"))
    db.close()
    out = {}
    for e in EMOZIONI:
        f = (z["stati"][[pos[i] for i, t in tag.items() if t == e and i in pos]] - z["mu"]).mean(0)
        out[e] = f / np.linalg.norm(f)
    return out


def stato(testo):
    return np.asarray(ponte.leggi_sonda(testo)["mean"])


FACCE = facce()


def kl(p, q):
    chiavi = set(p) | set(q)
    return float(sum(p.get(k, 1e-6) * np.log(p.get(k, 1e-6) / q.get(k, 1e-6)) for k in chiavi))


ris = []
for c in json.load(open(f"{DATI}/contesti.json")):
    ponte.calma()
    base, bt, bp = completa(c["testo"])
    sb = stato(base)
    print(f"== {c['nome']} ({c['n_token']} token) base: {base[:90]!r}", flush=True)
    for e in EMOZIONI:
        for f in DOSI:
            ponte.emozione(e, f)
            try:
                t, tt, tp = completa(c["testo"])
            finally:
                ponte.calma()
            div = next((i for i, (a, b) in enumerate(zip(bt, tt)) if a != b), min(len(bt), len(tt)))
            diversi = sum(a != b for a, b in zip(bt, tt)) / max(1, min(len(bt), len(tt)))
            d = stato(t) - sb
            d /= np.linalg.norm(d) + 1e-9
            eco = {x: float(d @ FACCE[x]) for x in EMOZIONI}
            r = {"eco": eco, "giusta": diversi > 0 and max(eco, key=eco.get) == e, "contesto": c["nome"], "n_token": c["n_token"], "emozione": e, "dose": f,
                 "kl_primo": kl(tp, bp), "prima_divergenza": div, "quota_diversi": diversi, "testo": t, "base": base}
            ris.append(r)
            print(f"   {e} x{f}: KL {r['kl_primo']:.3f}, diverge al token {div}, diversi {diversi:.2f}, "
                  f"eco {eco[e]:+.2f} {'giusta' if r['giusta'] else 'NO ' + max(eco, key=eco.get)}: {t[:70]!r}", flush=True)
            json.dump(ris, open(f"{D}/risultati-steering.json", "w"), ensure_ascii=False, indent=1)

print("\n| contesto | token | KL primo token (media) | prima divergenza (media) | quota token diversi (media) | emozione giusta |")
print("|---|---|---|---|---|---|")
for nome in dict.fromkeys(r["contesto"] for r in ris):
    rr = [r for r in ris if r["contesto"] == nome]
    print(f"| {nome} | {rr[0]['n_token']} | {np.mean([r['kl_primo'] for r in rr]):.3f} | "
          f"{np.mean([r['prima_divergenza'] for r in rr]):.1f} | {np.mean([r['quota_diversi'] for r in rr]):.2f} | "
          f"{sum(r['giusta'] for r in rr)}/{len(rr)} |")
