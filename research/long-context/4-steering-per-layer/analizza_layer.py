#!/usr/bin/env python3
"""Analisi di steering-layer.jsonl (12/10/2026): scheda per ognuna delle 51 emozioni + pagina di riepilogo per
la dashboard (http://<dashboard>/emozioni/). Funziona anche a prova in corso, sui punti gia' misurati.
Punto EFFICACE = testo cambiato, niente parola intrusa, non rotto, faccia giusta fra le prime 3 delle 51.
Per emozione e layer: efficacia = quota di contesti con almeno una dose efficace; per contesto la dose migliore e'
quella efficace con eco piu' alta, il range efficace va dalla prima all'ultima dose efficace.
Layer migliore = efficacia piu' alta (a pari, eco media piu' alta). Intensita' = mediana delle dosi migliori dei
contesti sul layer migliore; la scheda mostra le dosi intensita' +-0.10 su quel layer, contesto per contesto.
Uscite: emozioni-layer/dati.json, emozioni-layer/index.html, emozioni-layer/schede/<emozione>.md"""
import json
import os
import statistics as st
from collections import defaultdict

D = "/data/memoria-episodica-affettiva/research/long-context/4-steering-per-layer"
DATI = "/data/memoria-episodica-affettiva/research/long-context/dati"
OUT = f"{D}/emozioni-layer"
LAYER = list(range(15, 33))


def efficace(r):
    return r["cambiato"] and not r["parola"] and not r["rotto"] and (r.get("rango") or 99) <= 3


def main():
    basi = {}   # k -> testo senza vettore
    grezzi, chiuse, rif = defaultdict(dict), set(), defaultdict(dict)   # una serie conta solo se chiusa ("fine"); alla ripresa l'ultima si rifa'
    for riga in open(f"{D}/steering-layer.jsonl"):
        r = json.loads(riga)
        if r.get("base"):
            basi[r["k"]] = r["testo"]
        elif r.get("fine"):
            chiuse.add((r["emo"], r["layer"], r["k"]))
        elif r.get("fine_rifinitura"):
            pass
        elif r.get("rifinitura"):   # 12/10: dosi a passi di 0.001 attorno ai range, si aggiungono alla serie
            rif[(r["emo"], r["layer"], r["k"])][r["dose"]] = r
        else:
            g = grezzi[(r["emo"], r["layer"], r["k"])]
            if g and r["dose"] <= max(g):   # la serie ricomincia (ripresa dopo un'interruzione): vale la nuova
                g.clear()
            g[r["dose"]] = r
    punti = {c: [w[d] for d in sorted(w)] for c, v in grezzi.items() if c in chiuse for w in [{**v, **rif[c]}]}   # (emo, layer, k) -> punti per dose
    ctxs = {c["k"]: c for c in json.load(open(f"{D}/contesti-layer.json"))}
    emozioni = sorted({e for e, _, _ in punti})
    dati = {"layer": LAYER, "contesti": [ctxs[k] for k in sorted(ctxs)], "emozioni": {},
            "it": {x["en"]: x["it"] for x in json.load(open("/data/memoria-episodica-affettiva/emozioni.json"))}}
    for e in emozioni:
        per_layer = {}
        for L in LAYER:
            ks = sorted(k for (ee, ll, k) in punti if ee == e and ll == L)
            if not ks:
                continue
            migliori, inizi, fini, eco = [], [], [], []
            for k in ks:
                ef = [r for r in punti[(e, L, k)] if efficace(r)]
                if ef:
                    b = max(ef, key=lambda r: r["eco"])
                    migliori.append(b["dose"])
                    eco.append(b["eco"])
                    inizi.append(min(r["dose"] for r in ef))
                    fini.append(max(r["dose"] for r in ef))
            rotti = [next((r["dose"] for r in punti[(e, L, k)] if r["rotto"]), None) for k in ks]
            per_layer[L] = {
                "efficacia": round(len(migliori) / len(ks), 3), "n_contesti": len(ks),
                "dose": round(st.median(migliori), 2) if migliori else None,
                "dev": round(st.pstdev(migliori), 3) if len(migliori) > 1 else None,
                "da": round(st.median(inizi), 2) if inizi else None, "a": round(st.median(fini), 2) if fini else None,
                "eco": round(st.mean(eco), 3) if eco else None,
                "rotto_da": round(st.median([x for x in rotti if x is not None]), 2) if any(x is not None for x in rotti) else None}
        if not per_layer:
            continue
        Lb = max(per_layer, key=lambda L: (per_layer[L]["efficacia"], per_layer[L]["eco"] or -1))
        b = per_layer[Lb]
        finestra = []
        if b["dose"] is not None:
            for k in sorted(k for (ee, ll, k) in punti if ee == e and ll == Lb):
                per_dose = {r["dose"]: r for r in punti[(e, Lb, k)]}
                celle = []
                for i in range(-5, 6):
                    d = round(round(b["dose"] / 0.02) * 0.02 + 0.02 * i, 2)   # la tabella resta sulla scala a 0.02
                    r = per_dose.get(d)
                    celle.append(None if r is None else {"dose": d, "rango": r.get("rango"), "eco": round(r.get("eco", 0), 3),
                                                         "efficace": efficace(r), "parola": r["parola"], "rotto": r["rotto"],
                                                         "cambiato": r["cambiato"], "testo": r["testo"][:200]})
                finestra.append({"k": k, "n_token": ctxs[k]["n_token"], "celle": celle})
        confronto = []   # per contesto: testo senza vettore e col vettore al layer migliore, all'intensita' scelta
        if b["dose"] is not None:
            for k in sorted(k for (ee, ll, k) in punti if ee == e and ll == Lb):
                sotto = [r for r in punti[(e, Lb, k)] if r["dose"] <= b["dose"] + 1e-9]
                if sotto:
                    r = sotto[-1]
                    confronto.append({"k": k, "n_token": ctxs[k]["n_token"], "senza": basi.get(k), "con": r["testo"],
                                      "dose": r["dose"], "rango": r.get("rango"), "efficace": efficace(r),
                                      "parola": r["parola"], "rotto": r["rotto"]})
        dati["emozioni"][e] = {"layer": per_layer, "migliore": Lb, "finestra": finestra, "confronto": confronto}
        scheda(e, dati["emozioni"][e], ctxs, dati["it"][e])
    # 12/10: cosa e' in prova adesso: contesto dall'ultimo "== contesto" del log, emozione/layer/dose dall'ultimo punto
    kk = [int(r.split()[2]) for r in open(f"{D}/log-layer.txt") if r.startswith("== contesto")]
    ultimo = json.loads(open(f"{D}/steering-layer.jsonl").readlines()[-1])
    if kk:
        dati["in_corso"] = {"k": kk[-1], "n_token": ctxs[kk[-1]]["n_token"], "emo": ultimo["emo"],
                            "layer": ultimo["layer"], "dose": ultimo.get("dose")}
    os.makedirs(OUT, exist_ok=True)
    json.dump(dati, open(f"{OUT}/dati.json", "w"), ensure_ascii=False)
    html = open(f"{D}/pagina-emozioni.html").read()
    open(f"{OUT}/index.html", "w").write(html)
    print(f"{len(dati['emozioni'])} emozioni analizzate -> {OUT}/", flush=True)


def scheda(e, x, ctxs, nome_it):
    os.makedirs(f"{OUT}/schede", exist_ok=True)
    Lb, b = x["migliore"], x["layer"][x["migliore"]]
    righe = [f"# {nome_it} ({e})", "",
             f"Layer migliore: **L{Lb}**, efficace in {b['efficacia']:.0%} dei contesti; intensita' {b['dose']} "
             f"(dev. std fra contesti {b['dev']}), range efficace {b['da']}-{b['a']}, testo rotto da {b['rotto_da']}.", "",
             "| layer | efficacia | intensita' | dev | range efficace | eco | rotto da |", "|---|---|---|---|---|---|---|"]
    for L, v in sorted(x["layer"].items()):
        righe.append(f"| L{L} | {v['efficacia']:.0%} | {v['dose']} | {v['dev']} | {v['da']}-{v['a']} | {v['eco']} | {v['rotto_da']} |")
    if x["finestra"]:
        dosi = [round(b["dose"] + 0.02 * i, 2) for i in range(-5, 6)]
        righe += ["", f"## L{Lb}, intensita' {b['dose']} +-0.10 (rango della faccia giusta; * = efficace, P = parola, R = rotto)", "",
                  "| contesto | " + " | ".join(str(d) for d in dosi) + " |", "|---|" + "---|" * len(dosi)]
        for f in x["finestra"]:
            cel = [("-" if c is None else f"{c['rango'] or '='}{'*' if c['efficace'] else ''}{'P' if c['parola'] else ''}"
                    f"{'R' if c['rotto'] else ''}") for c in f["celle"]]
            righe.append(f"| {f['n_token']} | " + " | ".join(cel) + " |")
    open(f"{OUT}/schede/{e}.md", "w").write("\n".join(righe) + "\n")


if __name__ == "__main__":
    main()
