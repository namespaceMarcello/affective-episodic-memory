#!/usr/bin/env python3
"""Vettori emotivi in ITALIANO (12/10/2026): contrastivi sul dataset diaristico italiano.
Fonte: /data/dataset-codec/io-io-it-v2 (monologo interiore) e io-tu-it-v2 (parlato a un tu), ~300 testi per
ognuna delle 51 emozioni, senza la parola dell'emozione dentro (controllo-qualita.py): il vettore porta lo stato,
non la parola. Sostituiscono emovec-18 (gradiente sulla parola INGLESE, prompt Wikipedia inglesi) ed emovec-rossi.
Lettura: /lux-read del server di prova con "layers" (12/10: tutti i layer L15-L32 in una passata), stati GREZZI
(niente lente: lo steering agisce sul residuo grezzo), media sui token del testo.
Vettore: v_e(l) = media dei testi di e - media delle medie delle altre 50 emozioni (bilanciato per emozione).
Uscite: stati-diario.npy (testi x 18 x 2048, float16, si riprende da dove era), indice-diario.json,
/data/jspace/out-occamy/emovec-it.pt (stesso formato di emovec-rossi: {"vettori": {e: {l: tensore}}}).
Uso: lancia.sh emovec"""
import glob
import json
import os
import time
import urllib.request

import numpy as np
import torch

D = "/data/memoria-episodica-affettiva/research/long-context/5-vettori-italiani"
URL = os.environ.get("PONTE_URL", "http://127.0.0.1:8099")
OUT35 = "/data/jspace/out-occamy"
LAYER = list(range(15, 33))
D_MODEL = 2048

testi = []
for f in sorted(glob.glob("/data/dataset-codec/io-io-it-v2/*.jsonl") + glob.glob("/data/dataset-codec/io-tu-it-v2/*.jsonl")):
    registro = f.split("/")[-2]
    for riga in open(f):
        if riga.strip():
            r = json.loads(riga)
            testi.append({"emozione": r["emozione"], "registro": registro, "testo": r["testo"]})
json.dump([{k: v for k, v in t.items() if k != "testo"} for t in testi], open(f"{D}/indice-diario.json", "w"))

F = f"{D}/stati-diario.npy"
FATTI = f"{D}/fatti.txt"
stati = np.lib.format.open_memmap(F, mode="r+" if os.path.exists(F) else "w+", dtype=np.float16,
                                  shape=(len(testi), len(LAYER), D_MODEL))
fatti = int(open(FATTI).read()) if os.path.exists(FATTI) else 0
print(f"{len(testi)} testi, riparto da {fatti}", flush=True)
t0 = time.time()
for i in range(fatti, len(testi)):
    body = {"content": testi[i]["testo"], "layer": LAYER[0], "layers": LAYER[1:]}
    req = urllib.request.Request(URL + "/lux-read", json.dumps(body).encode(), {"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=600))
    righe = [r["mean"]] + [r["means"][str(l)] for l in LAYER[1:]]
    m = np.asarray(righe, np.float32)
    assert m.shape == (len(LAYER), D_MODEL) and np.isfinite(m).all(), f"testo {i}: stati non validi"
    stati[i] = m.astype(np.float16)
    if (i + 1) % 100 == 0 or i + 1 == len(testi):
        stati.flush()
        open(FATTI, "w").write(str(i + 1))
        v = (i + 1 - fatti) / (time.time() - t0)
        print(f"   {i + 1}/{len(testi)}  {v:.2f} testi/s, mancano {(len(testi) - i - 1) / v / 60:.0f} min", flush=True)

# vettori contrastivi
nomi = [x["en"] for x in json.load(open("/data/memoria-episodica-affettiva/emozioni.json"))]
etichette = np.array([t["emozione"] for t in testi])
medie = {e: stati[etichette == e].astype(np.float32).mean(0) for e in nomi}   # [18, 2048]
vettori = {}
for e in nomi:
    altre = np.mean([medie[x] for x in nomi if x != e], axis=0)
    vettori[e] = {l: torch.from_numpy(medie[e][j] - altre[j]) for j, l in enumerate(LAYER)}
torch.save({"vettori": vettori, "banda": LAYER, "metodo": "contrastivo-diario-it", "data": "2026-10-12",
            "fonte": "dataset-codec io-io-it-v2 + io-tu-it-v2", "n_testi": {e: int((etichette == e).sum()) for e in nomi}},
           f"{OUT35}/emovec-it.pt")
print(f"scritto {OUT35}/emovec-it.pt: {len(vettori)} emozioni x {len(LAYER)} layer", flush=True)
