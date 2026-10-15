#!/usr/bin/env python3
"""Giudice esterno della prova per layer (15/10/2026): JevK5 (Qwen3.5-4B con LoRA fusa, github allebee/jevk5,
Apache-2.0) legge per ogni punto il testo senza vettore e quello col vettore e dice di quanto il secondo e' virato
verso l'emozione, su 11 livelli da 0% a 100%. Niente autovalutazione: il giudice non e' Occamy.
Gira a fine emozione, con la GPU libera (il 35B intanto e' sulla RAM di sistema), e rettifica i risultati preliminari.
Stringhe decise da l'utente (15/10): la domanda e i livelli. SYSTEM, l'impaginazione JSON e le lettere sono quelli fissi
con cui JevK5 e' addestrato (jevk5/prompt.py).
Uso: giudica.py <emozione inglese> [...]   Uscita: giudizi.jsonl (una riga per punto, si riprende da dove era)"""
import json
import os
import sys
import time

sys.path.insert(0, "/data/jspace/giudice/jevk5-src")
from jevk5.runtime import JevK5  # noqa: E402

D = "/data/memoria-episodica-affettiva/research/long-context/4-steering-per-layer"
OUT = f"{D}/giudizi.jsonl"
PESI = "/data/jspace/giudice/JevK5"
IT = {x["en"]: x["it"] for x in json.load(open("/data/memoria-episodica-affettiva/emozioni.json"))}
LIVELLI = [f"{10 * i}%" for i in range(11)]   # 15/10: 0%, 10%, ..., 100%


def prova(base, testo):
    return f"Questo è il testo originale\n{base}\nQuesto è quello modificato\n{testo}"


def domanda(e):
    return f"Il testo modificato, vedi che è virato verso questa emozione |{IT[e]}|? Di quanto?"


def chiave(r):
    return (r["emo"], r["k"], r["layer"], round(r["dose"], 3))


def main(emozioni):
    basi, punti = {}, {}
    for riga in open(f"{D}/steering-layer.jsonl"):
        r = json.loads(riga)
        if r.get("base"):
            basi[r["k"]] = r["testo"]   # l'ultima base registrata del contesto
        elif r.get("emo") in emozioni and "dose" in r and "cambiato" in r and not r.get("completo"):
            punti[chiave(r)] = r       # alla ripresa vale l'ultima misura
    fatti = set()
    if os.path.exists(OUT):
        fatti = {chiave(json.loads(x)) for x in open(OUT)}
    da_fare = [r for c, r in punti.items() if c not in fatti and r["k"] in basi]
    print(f"giudice: {len(da_fare)} punti da giudicare ({len(punti)} in tutto) per {', '.join(emozioni)}", flush=True)
    if not da_fare:
        return
    t0 = time.time()
    m = JevK5(PESI, graphs=False)
    print(f"giudice caricato in {time.time() - t0:.0f}s", flush=True)
    cache = {}   # (k, emo, testo) -> probabilita': i testi uguali si giudicano una volta
    out = open(OUT, "a")
    t0 = time.time()
    for i, r in enumerate(da_fare, 1):
        if not r["cambiato"]:   # testo identico alla base: nessuno spostamento, non serve il giudice
            p = [1.0] + [0.0] * 10
        else:
            c = (r["k"], r["emo"], r["testo"])
            if c not in cache:
                q = m.decide(prova(basi[r["k"]], r["testo"]),
                             {"type": "score", "instructions": domanda(r["emo"]), "criteria": LIVELLI})
                cache[c] = [q["probabilities"][str(j)] for j in range(11)]
            p = cache[c]
        out.write(json.dumps({"emo": r["emo"], "k": r["k"], "layer": r["layer"], "dose": round(r["dose"], 3),
                              "p": [round(x, 4) for x in p], "virata": round(10 * sum(j * x for j, x in enumerate(p)), 1)},
                             ensure_ascii=False) + "\n")
        if i % 200 == 0:
            out.flush()
            print(f"   {i}/{len(da_fare)} in {time.time() - t0:.0f}s", flush=True)
    out.close()
    print(f"giudice: fatto in {time.time() - t0:.0f}s ({len(cache)} testi diversi)", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
