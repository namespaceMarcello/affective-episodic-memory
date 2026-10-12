#!/usr/bin/env python3
"""Prova contesto lungo, parte 1: la J-lens a L29 regge con 140k token di contesto vero? (11/10/2026)
Per ogni contesto di contesti.json legge dal server di prova (luxifer-prova, /lux-read lungo) gli stati degli
ULTIMI 512 token (uguali in tutti i contesti: finiscono tutti sullo stesso messaggio) a L29 e a L38.
La lente (lente35b.py) porta L29 nello spazio di L38 (TARGET = penultimo strato), quindi:
  lente = lm_head(norm(J29 @ h29))    quello che la lente legge a L29
  L38   = lm_head(norm(h38))          quello che c'e' davvero a L38 (il bersaglio della lente)
A L29 la lente legge CONCETTI, non la parola successiva (11/10: su un token
legge i concetti intorno, anche in altre lingue, non il token che segue), quindi il top-1 sul token vero non e' la misura. Misure per contesto:
  fedelta'   KL(L38 || lente) e sovrapposizione dei top-10 lente/L38
  stabilita' sovrapposizione dei top-10 della lente fra il contesto e il contesto isolato, stesso token
  controllo  top-1 di L38 sul token vero
  deriva     coseno fra gli stati L29 (lente meno base, lo spazio del richiamo semantico) dello stesso token
             nel contesto e nel contesto isolato, per token e sulla media della finestra
  campioni   le prime 5 parole lette dalla lente su alcuni token, da leggere
Uscita: risultati-lente.json e una tabella su stdout."""
import json
import os
import sys
import urllib.request

import numpy as np
import transformers
from safetensors import safe_open

sys.path.insert(0, "/data/memoria-episodica-affettiva")
import ponte  # noqa: E402

URL = os.environ.get("PROVA_URL", "http://127.0.0.1:8099")
D = "/data/memoria-episodica-affettiva/research/long-context/1-lente"
DATI = "/data/memoria-episodica-affettiva/research/long-context/dati"
HF = "/data/jspace/hf/hub/models--Accio-Lab--occamy-1.0/snapshots/f9e2771699f14d9c4bbbc0c7b913e58ebe1442d0"
ULTIMI = 512
TOK = transformers.AutoTokenizer.from_pretrained("Accio-Lab/occamy-1.0")


def tensore(nome):
    m = json.load(open(f"{HF}/model.safetensors.index.json"))["weight_map"]
    with safe_open(f"{HF}/{m[nome]}", "pt") as f:
        return f.get_tensor(nome).float().numpy()


W = tensore("lm_head.weight")                        # [vocab, 2048]
G = 1.0 + tensore("model.language_model.norm.weight")   # RMSNorm di Qwen3.5-MoE: x * (1 + w)


def logit(h):
    h = h / np.sqrt((h * h).mean(-1, keepdims=True) + 1e-6) * G
    return h @ W.T


def logsoftmax(x):
    x = x - x.max(-1, keepdims=True)
    return x - np.log(np.exp(x).sum(-1, keepdims=True))


def leggi(testo, layer):
    body = {"content": testo, "layer": layer, "per_token": True, "ultimi": ULTIMI, "lungo": True, "speciali": True}
    r = urllib.request.Request(URL + "/lux-read", json.dumps(body).encode(), {"Content-Type": "application/json"})
    d = json.load(urllib.request.urlopen(r, timeout=7200))
    return np.asarray(d["states"], np.float32), d["token_ids"], d["n_tokens"]


def top(lp, k=10):
    return np.argpartition(-lp, k, axis=-1)[:, :k]


def jaccard(A, B):
    return float(np.mean([len(set(a) & set(b)) / len(set(a) | set(b)) for a, b in zip(A, B)]))


base = ponte.base_lente()
contesti = json.load(open(f"{DATI}/contesti.json"))
ris, rif = [], None
for c in contesti:
    h29, ids, ntot = leggi(c["testo"], 29)
    h38, ids38, _ = leggi(c["testo"], 38)
    assert ids == ids38
    lens, l38 = [], []
    for a in range(0, len(ids), 64):   # a blocchi: il vocabolario e' grande
        lens.append(logsoftmax(logit(ponte.trasporta(h29[a:a + 64]))))
        l38.append(logsoftmax(logit(h38[a:a + 64])))
    lens, l38 = np.concatenate(lens), np.concatenate(l38)
    vero = np.asarray(ids[1:])
    tl, t38 = top(lens), top(l38)
    q = ponte.trasporta(h29) - base
    q /= np.linalg.norm(q, axis=1, keepdims=True) + 1e-9
    r = {"nome": c["nome"], "n_token": ntot,
         "kl_l38_lente": float((np.exp(l38) * (l38 - lens)).sum(-1).mean()),
         "top10_lente_l38": jaccard(tl, t38),
         "l38_top1_vero": float((l38[:-1].argmax(-1) == vero).mean()),
         "campioni": {TOK.decode([ids[i]]): [TOK.decode([int(x)]) for x in lens[i].argsort()[::-1][:5]]
                      for i in range(40, len(ids), 80)}}
    if rif is None:   # il primo e' "isolato": il riferimento di stabilita' e deriva
        rif = (ids, q, tl)
    if ids == rif[0]:
        r["stabilita_top10"] = jaccard(tl, rif[2])
        r["deriva_token"] = float((q * rif[1]).sum(-1).mean())
        m1, m0 = q.mean(0), rif[1].mean(0)
        r["deriva_media_finestra"] = float(m1 @ m0 / (np.linalg.norm(m1) * np.linalg.norm(m0)))
    ris.append(r)
    print(json.dumps(r, ensure_ascii=False), flush=True)
    json.dump(ris, open(f"{D}/risultati-lente.json", "w"), indent=1, ensure_ascii=False)

print("\n| contesto | token | KL L38/lente | top-10 lente vs L38 | stabilita' lente (vs isolato) | L38 top-1 vero | deriva per token | deriva media |")
print("|---|---|---|---|---|---|---|---|")
for r in ris:
    print(f"| {r['nome']} | {r['n_token']} | {r['kl_l38_lente']:.2f} | {r['top10_lente_l38']:.3f} | {r.get('stabilita_top10', float('nan')):.3f} | "
          f"{r['l38_top1_vero']:.3f} | {r.get('deriva_token', float('nan')):.3f} | {r.get('deriva_media_finestra', float('nan')):.3f} |")
