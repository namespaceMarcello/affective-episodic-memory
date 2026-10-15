#!/usr/bin/env python3
"""13/10: vettori rifatti sui passaggi nuovi in prima persona per le emozioni che al contesto 0 della prova per
layer non hanno avuto nessuna dose efficace (i passaggi vecchi in passaggi/<nome>.txt.bak-terza-persona).
Rilegge i loro stati, li sostituisce in stati-template.npy, ricalcola la template lens e in vettori-paper.pt cambia
SOLO queste emozioni (le altre template restano quelle del 12/10).
Paura e' una parola di un token (J-lens) ma la J-lens non ha funzionato: passa alla template lens.
Il server e' una seconda istanza su CPU (aggiorna_emozioni.sh): quello della prova per layer ha un solo slot."""
import os
import shutil

import numpy as np
import torch

os.environ.setdefault("PONTE_URL", "http://127.0.0.1:8098")
import estrai_vettori as V

# 13/10 pomeriggio: secondo giro, frasi che chiamano la parola con forza (stile dell'utente), per tutte le emozioni
# senza range utile al contesto 0 (nessuna dose efficace o 1-3 dosi isolate). Backup con SUFFISSO .bak-prima-persona-1
DA_RIFARE = ["acceptance", "amazement", "pride"]   # 13/10 sera: efficaci su un solo layer, un tentativo
SUFFISSO = ".bak-prima-persona-1"

testi = V.passaggi()
f = f"{V.D}/stati-template.npy"
if not os.path.exists(f + SUFFISSO):
    shutil.copy(f, f + SUFFISSO)
S = np.load(f)
for e in DA_RIFARE:
    idx = [i for i, (x, _) in enumerate(testi) if x == e]
    assert len(idx) == 30, e
    for i in idx:
        for j, L in enumerate(V.LAYER):
            S[i, j] = V.leggi(testi[i][1], L)
    np.save(f, S)
    print(f"   {V.ITALIANO[e]}: 30 passaggi letti", flush=True)

tl = V.template(S, np.array([e for e, _ in testi]))
if not os.path.exists(V.OUT + SUFFISSO):
    shutil.copy(V.OUT, V.OUT + SUFFISSO)
d = torch.load(V.OUT, weights_only=False)
for e in DA_RIFARE:
    vecchio = d["vettori"][e]
    d["vettori"][e] = d["template"][e] = tl[e]
    d["metodo"][e] = "template"
    print(f"   {V.ITALIANO[e]}: coseno vecchio/nuovo " +
          " ".join(f"L{L} {float(vecchio[L] @ tl[e][L]):.2f}" for L in (15, 24, 29, 32)), flush=True)
d.setdefault("prima_persona_giri", []).append({"data": "2026-10-13", "emozioni": DA_RIFARE})
torch.save(d, V.OUT)
print(f"salvato {V.OUT}", flush=True)
