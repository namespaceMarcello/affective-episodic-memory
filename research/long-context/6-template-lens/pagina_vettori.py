#!/usr/bin/env python3
"""Pagina dei vettori emotivi col metodo del paper workspace (12/10/2026): passaggi italiani, stato all'ultima
posizione, J-lens per le parole di un token e template lens per le altre. Solo CPU: legge stati-template.npy (anche a
estrazione in corso) e vettori-paper.pt quando c'e'. Uscita: vettori-pagina/dati.json + index.html; dashboard /vettori/
(si rigenera da sola quando l'estrazione ha dati nuovi)."""
import json
import os
import shutil

import numpy as np
import torch

D = "/data/memoria-episodica-affettiva/research/long-context/6-template-lens"
OUT = f"{D}/vettori-pagina"
LAYER = list(range(15, 33))
EMO = json.load(open("/data/memoria-episodica-affettiva/emozioni.json"))
NOMI = [x["en"] for x in EMO]
VP = "/data/jspace/out-occamy/vettori-paper.pt"
PRIMA = "/data/jspace/out-occamy/emovec-it.pt"   # i contrastivi sul diario (12/10 pomeriggio), solo per confronto

vec = lambda d, e, l: np.asarray(d[e][l], np.float32).ravel()
cos = lambda a, b: float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def main():
    os.makedirs(OUT, exist_ok=True)
    testi = []
    for x in EMO:
        testi += [(NOMI.index(x["en"]), r.strip()) for r in open(f"{D}/passaggi/{x['it']}.txt", encoding="utf-8") if r.strip()]
    fatti = int(open(f"{D}/fatti.txt").read()) if os.path.exists(f"{D}/fatti.txt") else 0
    dati = {"emo": [{"en": x["en"], "it": x["it"], "col": x["colore"], "emoji": x["emoji"]} for x in EMO], "layer": LAYER,
            "n_passaggi": len(testi), "letti": fatti, "testi": [t for _, t in testi[:fatti]],
            "etichette": [i for i, _ in testi[:fatti]], "pronti": os.path.exists(VP)}
    emo_lette = sorted(set(dati["etichette"]))
    if fatti >= 60 and len(emo_lette) >= 2:
        try:   # l'estrazione riscrive il file ogni 30 passaggi: se lo becchiamo a meta', resta la pagina di prima
            S = np.load(f"{D}/stati-template.npy")[:fatti]
        except Exception as x:
            print("stati-template.npy in scrittura:", x)
            return
        et = np.array(dati["etichette"])
        sep, nuvole = [], {}
        for j, l in enumerate(LAYER):   # rapporto di Fisher sui passaggi letti, come nella pagina dei vettori del diario
            X = np.asarray(S[:, j, :], np.float32)
            X = X - X.mean(0)
            M = np.stack([X[et == i].mean(0) for i in emo_lette])
            fra = float((M ** 2).sum(1).mean())
            dentro = float(np.mean([((X[et == i] - M[k]) ** 2).sum(1).mean() for k, i in enumerate(emo_lette)]))
            sep.append(round(fra / dentro, 4))
            if l in (15, 20, 24, 29, 32):
                _, _, Vt = np.linalg.svd(M - M.mean(0), full_matrices=False)
                P, C = X @ Vt[:2].T, M @ Vt[:2].T
                s = np.abs(P).max(0)
                nuvole[l] = {"x": np.round(P[:, 0] / s[0], 3).tolist(), "y": np.round(P[:, 1] / s[1], 3).tolist(),
                             "c": {int(i): [round(float(C[k, 0] / s[0]), 3), round(float(C[k, 1] / s[1]), 3)]
                                   for k, i in enumerate(emo_lette)}}
        dati["separazione"], dati["nuvole"] = sep, nuvole
    if dati["pronti"]:
        vp = torch.load(VP, weights_only=False)
        V, T = vp["vettori"], vp["template"]
        dati["metodo"] = [vp["metodo"][e] for e in NOMI]
        dati["data"] = vp["data"]
        dati["somiglianza"] = {}
        for l in LAYER:
            X = np.stack([vec(V, e, l) for e in NOMI])
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            dati["somiglianza"][l] = np.round(X @ X.T, 2).tolist()
        dati["jlens_vs_template"] = {e: [round(cos(vec(V, e, l), vec(T, e, l)), 3) for l in LAYER]
                                     for e in NOMI if vp["metodo"][e] == "jlens"}
        if os.path.exists(PRIMA):
            P = torch.load(PRIMA, weights_only=False)["vettori"]
            dati["vs_prima"] = [[round(cos(vec(V, e, l), vec(P, e, l)), 3) for l in LAYER] for e in NOMI]
            perm = np.random.default_rng(0).permutation(len(NOMI))
            dati["vs_prima_caso"] = [round(float(np.mean([cos(vec(V, NOMI[i], l), vec(P, NOMI[perm[i]], l))
                                                          for i in range(len(NOMI))])), 3) for l in LAYER]
    json.dump(dati, open(f"{OUT}/dati.json", "w"), ensure_ascii=False, separators=(",", ":"))
    shutil.copy(f"{D}/pagina-vettori.html", f"{OUT}/index.html")
    print(f"scritto {OUT} ({fatti}/{len(testi)} passaggi, vettori {'pronti' if dati['pronti'] else 'in calcolo'})")


if __name__ == "__main__":
    main()
