#!/usr/bin/env python3
"""Prova steering in PyTorch bf16 (12/10/2026): i vettori emotivi di emo-cvec-v2 spostano Occamy nel loader HF
(mixed35b, esperti su CPU), cioe' dove sono stati tarati il 28/07 (scan-c2: h += alpha * ||h|| * dir)?
Stessi contesti veri della prova sul server (isolato, sys), stesse emozioni, stesse dosi (centro * 0.5 / 1.0).
Per ogni dose due versioni:
  hf      h += dose * ||h|| * dir          (la taratura C2)
  server  h += dose * rms(h) * dir         (quello che fa luxifer col vettore che gli manda ponte.dirs_emozione:
                                            ||h|| = rms(h) * sqrt(2048), quindi dose / 45)
Misure come prova_steering.py: KL sul primo token, prima divergenza, quota di token diversi, testi completi.
Uscita: risultati-steering-hf.json e righe su stdout."""
import json
import sys

import numpy as np
import torch

sys.path.insert(0, "/data/jspace")
import mixed35b  # noqa: E402

D = "/data/memoria-episodica-affettiva/research/long-context/2-steering"
DATI = "/data/memoria-episodica-affettiva/research/long-context/dati"
EMOZIONI = ["joy", "sadness", "fear", "curiosity"]
DOSI = [0.5, 1.0]
N = 64

z = np.load("/data/memoria-episodica-affettiva/emo-cvec-v2.npz")
nomi = [str(n) for n in z["nomi"]]
s = mixed35b.load()
hf, tok, model = s["hf"], s["tok"], s["model"]
D_EMB = hf.config.get_text_config().hidden_size


def genera(testo, emo=None, fattore=1.0, scala=1.0):
    ids = tok(testo, return_tensors="pt", add_special_tokens=False).to("cuda")
    handles = []
    if emo is not None:
        i = nomi.index(emo)
        dose = min(float(z["centro"][i]) * fattore, float(z["bordo"][i])) * scala
        for k, l in enumerate(z["layers"][i]):
            v = torch.tensor(z["dirs"][i, k]).to("cuda", torch.bfloat16)

            def hook(module, args, kwargs, output, v=v):
                h = output[0] if isinstance(output, tuple) else output
                h = h + dose * h.norm(dim=-1, keepdim=True) * v
                return (h, *output[1:]) if isinstance(output, tuple) else h

            handles.append(model.layers[int(l)].register_forward_hook(hook, with_kwargs=True))
    try:
        with torch.no_grad():
            out = hf.generate(**ids, max_new_tokens=N, do_sample=False, temperature=None, top_p=None, top_k=None,
                              output_scores=True, return_dict_in_generate=True)
    finally:
        for h in handles:
            h.remove()
    nuovi = out.sequences[0, ids.input_ids.shape[1]:].tolist()
    return tok.decode(nuovi), nuovi, torch.log_softmax(out.scores[0][0].float(), -1).cpu()


ris = []
for c in json.load(open(f"{DATI}/contesti.json"))[:2]:   # isolato, sys
    base, bt, bl = genera(c["testo"])
    print(f"== {c['nome']} ({c['n_token']} token) base: {base[:90]!r}", flush=True)
    r = {"nome": c["nome"], "base": base, "prove": []}
    for e in EMOZIONI:
        for f in DOSI:
            for modo, scala in (("hf", 1.0), ("server", 1.0 / np.sqrt(D_EMB))):
                t, tt, tl = genera(c["testo"], e, f, scala)
                kl = float((tl.exp() * (tl - bl)).sum())
                div = next((i for i, (a, b) in enumerate(zip(bt, tt)) if a != b), min(len(bt), len(tt)))
                diversi = sum(a != b for a, b in zip(bt, tt)) / max(1, min(len(bt), len(tt)))
                print(f"   {e} x{f} {modo:6s}: KL {kl:.3f}, diverge al token {div}, diversi {diversi:.2f}: {t[:70]!r}",
                      flush=True)
                r["prove"].append({"emo": e, "fattore": f, "modo": modo, "kl": kl, "diverge": div,
                                   "diversi": diversi, "testo": t})
            json.dump(ris + [r], open(f"{D}/risultati-steering-hf.json", "w"), indent=1, ensure_ascii=False)
    ris.append(r)
