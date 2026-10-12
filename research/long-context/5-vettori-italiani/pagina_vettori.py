#!/usr/bin/env python3
"""Pagina grafica del lavoro sui vettori emotivi italiani (12/10/2026): dalla lettura dei testi diaristici ai
vettori contrastivi, e il confronto coi vettori inglesi. Solo CPU, legge stati-diario.npy ed emovec-*.pt.
Uscita: vettori-pagina/dati.json + index.html (copia di pagina-vettori.html); nella dashboard su /vettori/."""
import json
import shutil

import numpy as np
import torch

D = "/data/memoria-episodica-affettiva/research/long-context/5-vettori-italiani"
OUT = f"{D}/vettori-pagina"
LAYER = list(range(15, 33))
EMO = json.load(open("/data/memoria-episodica-affettiva/emozioni.json"))
NOMI = [x["en"] for x in EMO]

indice = json.load(open(f"{D}/indice-diario.json"))
etich = np.array([NOMI.index(t["emozione"]) for t in indice])
registro = np.array([t["registro"] for t in indice])
stati = np.load(f"{D}/stati-diario.npy", mmap_mode="r")
it = torch.load("/data/jspace/out-occamy/emovec-it.pt", weights_only=False)
en = torch.load("/data/jspace/out-occamy/emovec-18.pt", weights_only=False)["vectors"]
rossi = torch.load("/data/jspace/out-occamy/emovec-rossi.pt", weights_only=False)["vettori"]
vec = lambda d, e, l: np.asarray(d[e][l], np.float32).ravel()
cos = lambda a, b: float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

dati = {"emo": [{"en": x["en"], "it": x["it"], "col": x["colore"], "emoji": x["emoji"]} for x in EMO], "layer": LAYER,
        "n_testi": len(indice), "registri": {r: int((registro == r).sum()) for r in sorted(set(registro))},
        "per_emo": [[int(((etich == i) & (registro == r)).sum()) for r in sorted(set(registro))] for i in range(len(NOMI))],
        "data": it["data"], "metodo": it["metodo"]}

# separazione per layer: varianza fra emozioni / varianza dentro l'emozione (rapporto di Fisher, su tutte le dimensioni)
sep, nuvole = [], {}
rng = np.random.default_rng(0)
for j, l in enumerate(LAYER):
    X = np.asarray(stati[:, j, :], np.float32)
    X -= X.mean(0)
    M = np.stack([X[etich == i].mean(0) for i in range(len(NOMI))])
    fra = float((M ** 2).sum(1).mean())
    dentro = float(np.mean([((X[etich == i] - M[i]) ** 2).sum(1).mean() for i in range(len(NOMI))]))
    sep.append(round(fra / dentro, 4))
    if l in (15, 20, 24, 29, 32):   # nuvola dei testi: proiezione sui 2 assi principali delle 51 medie
        _, _, Vt = np.linalg.svd(M - M.mean(0), full_matrices=False)
        P = X @ Vt[:2].T
        P /= np.abs(P).max(0)
        nuvole[l] = {"x": np.round(P[:, 0], 3).tolist(), "y": np.round(P[:, 1], 3).tolist(),
                     "cx": np.round((M @ Vt[:2].T)[:, 0] / np.abs(X @ Vt[:2].T).max(0)[0], 3).tolist(),
                     "cy": np.round((M @ Vt[:2].T)[:, 1] / np.abs(X @ Vt[:2].T).max(0)[1], 3).tolist()}
    print(f"L{l}: separazione {sep[-1]}", flush=True)
dati["separazione"], dati["nuvole"], dati["etichette"] = sep, nuvole, etich.tolist()

# vettori: norma per layer, somiglianza fra emozioni, distanza dagli inglesi
V = it["vettori"]
dati["norma"] = [[round(float(np.linalg.norm(vec(V, e, l))), 3) for l in LAYER] for e in NOMI]
dati["somiglianza"] = {}
for l in LAYER:   # tutti i 18 layer (12/10: mancava L29)
    X = np.stack([vec(V, e, l) for e in NOMI]); X /= np.linalg.norm(X, axis=1, keepdims=True)
    dati["somiglianza"][l] = np.round(X @ X.T, 2).tolist()
dati["vs_en"] = [[round(cos(vec(V, e, l), vec(en, e, l)), 3) for l in LAYER] for e in NOMI]
perm = rng.permutation(len(NOMI))
dati["vs_en_caso"] = [round(float(np.mean([cos(vec(V, NOMI[i], l), vec(en, NOMI[perm[i]], l)) for i in range(len(NOMI))])), 3) for l in LAYER]
dati["vs_rossi"] = {e: [round(cos(vec(V, e, l), vec(rossi, e, l)), 3) for l in LAYER] for e in NOMI if e in rossi}
dati["stessa_vicina_L24"] = sum(max(NOMI, key=lambda f: cos(vec(V, e, 24), vec(en, f, 24))) == e for e in NOMI)

json.dump(dati, open(f"{OUT}/dati.json", "w"), ensure_ascii=False, separators=(",", ":"))
shutil.copy(f"{D}/pagina-vettori.html", f"{OUT}/index.html")
print("scritto", OUT)
