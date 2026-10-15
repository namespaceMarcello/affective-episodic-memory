#!/usr/bin/env python3
"""Analisi di steering-layer.jsonl (12/10/2026): scheda per ognuna delle 51 emozioni + pagina di riepilogo per
la dashboard (http://<dashboard>/emozioni/). Funziona anche a prova in corso, sui punti gia' misurati.
Punto EFFICACE = testo cambiato, niente parola intrusa, non rotto, faccia giusta fra le prime 4 delle 51 (fino al 15/10: 3).
Per emozione e layer: efficacia = quota di contesti con almeno una dose efficace; per contesto la dose migliore e'
quella efficace con eco piu' alta, il range efficace va dalla prima all'ultima dose efficace.
Layer migliore = efficacia piu' alta; a pari, range efficace piu' ampio (13/10 l'utente: conta il range, non l'eco;
l'eco decide solo a pari range). Intensita' = mediana delle dosi migliori dei
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


RANGO_MAX = 4   # 15/10: efficace = faccia giusta fra le prime 4 delle 51 (prima 3)


def efficace(r):
    return r["cambiato"] and not r["parola"] and not r["rotto"] and (r.get("rango") or 99) <= RANGO_MAX


def main():
    basi = {}   # k -> testo senza vettore
    liberi, basi_libere = {}, {}
    interi, basi_intere = {}, {}   # 13/10: risposte intere a fine contesto, (emo, k) -> riga; k -> testo
    grezzi, chiuse, rif = defaultdict(dict), set(), defaultdict(dict)   # una serie conta solo se chiusa ("fine"); alla ripresa l'ultima si rifa'
    for riga in open(f"{D}/steering-layer.jsonl"):
        r = json.loads(riga)
        if r.get("base"):
            basi[r["k"]] = r["testo"]
        elif r.get("base_completo"):
            basi_intere[r["k"]] = r["testo"]
        elif r.get("completo"):
            interi[(r["emo"], r["k"], r["layer_scelto"])] = r   # 15/10: una risposta per layer
        elif r.get("libero"):   # 15/10: risposta libera (vettore acceso, niente grammatica) con l'esame della tool call
            liberi[(r["emo"], r["k"], r["layer_scelto"])] = r
        elif r.get("base_libero"):
            basi_libere[r["k"]] = r
        elif r.get("fine"):
            chiuse.add((r["emo"], r["layer"], r["k"]))
        elif r.get("fine_rifinitura") or r.get("fine_risposte"):
            pass
        elif r.get("rifinitura"):   # 12/10: dosi a passi di 0.001 attorno ai range, si aggiungono alla serie
            rif[(r["emo"], r["layer"], r["k"])][r["dose"]] = r
        else:
            g = grezzi[(r["emo"], r["layer"], r["k"])]
            if g and r["dose"] <= max(g):   # la serie ricomincia (ripresa dopo un'interruzione): vale la nuova
                g.clear()
            g[r["dose"]] = r
    tutti = {c: [w[d] for d in sorted(w)] for c, v in grezzi.items() if c in chiuse for w in [{**v, **rif[c]}]}   # (emo, layer, k) -> punti per dose
    ctxs = {c["k"]: c for c in json.load(open(f"{D}/contesti-layer.json"))}
    it = {x["en"]: x["it"] for x in json.load(open("/data/memoria-episodica-affettiva/emozioni.json"))}
    # 15/10: una pagina per contesto (pulsanti C0 C1 ...), ognuna coi dati del suo solo contesto.
    # dati.json = indice piccolo (contesti disponibili, cosa e' in prova); dati-c<k>.json = i dati del contesto k
    kk = [int(r.split()[2]) for r in open(f"{D}/log-layer.txt") if r.startswith("== contesto")]
    ultimo = json.loads(open(f"{D}/steering-layer.jsonl").readlines()[-1])
    indice = {"contesti": [ctxs[k] for k in sorted(ctxs)], "it": it,
              "disponibili": sorted({k for _, _, k in tutti} | ({kk[-1]} if kk else set()))}
    if kk:
        indice["in_corso"] = {"k": kk[-1], "n_token": ctxs[kk[-1]]["n_token"], "emo": ultimo.get("emo"),
                              "layer": ultimo.get("layer", ultimo.get("layer_scelto")), "dose": ultimo.get("dose")}
    os.makedirs(OUT, exist_ok=True)
    riepilogo = {}   # 15/10: emozione -> {k: [layer efficaci, layer misurati]} per la pagina riepilogo
    for kc in indice["disponibili"]:
        punti = {c: v for c, v in tutti.items() if c[2] == kc}
        dati = dict(indice, layer=LAYER, k=kc, emozioni={}, base=basi.get(kc))   # base: testo senza vettore (15/10, per il riquadro)
        analizza(punti, dati, ctxs, basi, basi_intere, interi, kc)
        scrivi(f"{OUT}/dati-c{kc}.json", dati)
        for e, x in dati["emozioni"].items():
            riepilogo.setdefault(e, {})[kc] = [sum(v["efficacia"] > 0 for v in x["layer"].values()), len(x["layer"]),
                                               x["layer"][x["migliore"]]["dose"], x["migliore"],   # + dose e layer migliori del contesto
                                               [v["dose"] for v in x["layer"].values() if v["dose"] is not None]]   # dose migliore di ogni layer efficace
    indice["riepilogo"] = riepilogo
    # 15/10: per emozione e dose (scala a 0.02, su tutti i layer e i contesti): [punti efficaci, punti misurati]
    dosi = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for (e, _, _), pp in tutti.items():
        for r in pp:
            if abs(r["dose"] * 50 - round(r["dose"] * 50)) < 1e-6:   # fuori la rifinitura a passi di 0.001
                c = dosi[e][round(r["dose"], 2)]
                c[0] += efficace(r)
                c[1] += 1
    indice["dosi"] = {e: {str(d): v for d, v in sorted(x.items())} for e, x in dosi.items()}
    indice["n_layer"] = len(LAYER)
    scrivi(f"{OUT}/dati.json", indice)
    html = open(f"{D}/pagina-emozioni.html").read()
    open(f"{OUT}/index.html.tmp", "w").write(html)
    os.replace(f"{OUT}/index.html.tmp", f"{OUT}/index.html")
    open(f"{OUT}/riepilogo.html.tmp", "w").write(open(f"{D}/pagina-riepilogo.html").read())
    os.replace(f"{OUT}/riepilogo.html.tmp", f"{OUT}/riepilogo.html")
    print(f"{len(indice['disponibili'])} contesti analizzati -> {OUT}/", flush=True)


def scrivi(f, x):
    """15/10: scrittura atomica, la pagina non legge mai un file a meta'"""
    json.dump(x, open(f + ".tmp", "w"), ensure_ascii=False)
    os.replace(f + ".tmp", f)


def analizza(punti, dati, ctxs, basi, basi_intere, interi, kc):
    emozioni = sorted({e for e, _, _ in punti})
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
        def ampiezza(v):
            return v["a"] - v["da"] if v["da"] is not None else -1
        Lb = max(per_layer, key=lambda L: (per_layer[L]["efficacia"], ampiezza(per_layer[L]), per_layer[L]["eco"] or -1))
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
                                                         "cambiato": r["cambiato"], "testo": r["testo"],
                              "tool": r.get("tool_valida"), "tool_presente": r.get("tool_presente"),
                              "tool_errore": r.get("tool_errore"), "stop": r.get("stop")})
                finestra.append({"k": k, "n_token": ctxs[k]["n_token"], "celle": celle})
        confronto = []   # per contesto: testo senza vettore e col vettore al layer migliore, all'intensita' scelta
        if b["dose"] is not None:
            for k in sorted(k for (ee, ll, k) in punti if ee == e and ll == Lb):
                sotto = [r for r in punti[(e, Lb, k)] if r["dose"] <= b["dose"] + 1e-9]
                if sotto:
                    r = sotto[-1]
                    c = {"k": k, "n_token": ctxs[k]["n_token"], "senza": basi.get(k), "con": r["testo"],
                         "dose": r["dose"], "rango": r.get("rango"), "efficace": efficace(r),
                         "parola": r["parola"], "rotto": r["rotto"]}
                    if (e, k, Lb) in interi and k in basi_intere:   # risposta intera: vince sui 64 token
                        i = interi[(e, k, Lb)]
                        c.update(senza=basi_intere[k], con=i["testo"], intero=True, layer_intero=i["layer_scelto"],
                                 dose=i["dose"])
                    confronto.append(c)
        # 13/10: per contesto, una colonna per layer L15..L32: il punto all'intensita' di quel layer
        # (se il layer non ha dosi efficaci, il punto col rango migliore), col testo intero per il passaggio del mouse
        per_contesto = []
        for k in sorted({k for (ee, _, k) in punti if ee == e}):
            celle = []
            for L in LAYER:
                pp = punti.get((e, L, k))
                if not pp:
                    celle.append(None)
                    continue
                v = per_layer[L]
                r = (min(pp, key=lambda r: abs(r["dose"] - v["dose"])) if v["dose"] is not None
                     else min(pp, key=lambda r: (r.get("rango") or 99, -r.get("eco", 0))))
                celle.append({"layer": L, "dose": r["dose"], "rango": r.get("rango"), "eco": round(r.get("eco", 0), 3),
                              "efficace": efficace(r), "parola": r["parola"], "rotto": r["rotto"],
                              "cambiato": r["cambiato"], "testo": r["testo"],
                              "tool": r.get("tool_valida"), "tool_presente": r.get("tool_presente"),
                              "tool_errore": r.get("tool_errore"), "stop": r.get("stop")})
            per_contesto.append({"k": k, "n_token": ctxs[k]["n_token"], "celle": celle})
        dati["emozioni"][e] = {"layer": per_layer, "migliore": Lb, "finestra": finestra, "confronto": confronto,
                               "per_contesto": per_contesto}
        scheda(e, dati["emozioni"][e], ctxs, dati["it"][e], kc)


def scheda(e, x, ctxs, nome_it, kc):
    os.makedirs(f"{OUT}/schede/c{kc}", exist_ok=True)
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
    open(f"{OUT}/schede/c{kc}/{e}.md", "w").write("\n".join(righe) + "\n")


if __name__ == "__main__":
    main()
