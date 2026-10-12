#!/usr/bin/env python3
"""Vettori emotivi col metodo del paper Anthropic "Verbalizable Representations Form a Global Workspace in Language
Models" (12/10/2026: da zero, parole italiane), banda L15-L32.

- Parola italiana di UN token (" gioia"): J-lens vector = riga della parola in W_U J_l. Il paper decodifica con
  softmax(W_U norm(J_l h)); la norma finale di Occamy e' RMSNorm con peso (1 + w), quindi la riga e'
  J_l^T ((1 + w) * W_U[t]) (la RMSNorm toglie solo la scala).
- Parola di PIU' token (" ammir" + "azione"): template lens (appendice del paper). Passaggi in cui la parola e' la
  continuazione naturale (passaggi/<nome italiano>.txt), stato all'ultima posizione, media per parola, centrata sulla
  media delle altre parole e sbiancata = direzione discriminante lineare: (S_w + shrink)^-1 (mu_e - media delle altre),
  S_w = covarianza dentro le parole. Shrinkage verso l'identita' scalata (ALFA): con 1530 passaggi in 2048 dimensioni
  S_w da sola non si inverte. Il paper non dice come regolarizza: ALFA e' una scelta nostra.
Per le parole di un token si salva anche la template lens (confronto), ma il vettore usato e' il J-lens.
Uscite: stati-template.npy (passaggi x 18 x 2048, si riprende da fatti.txt), /data/jspace/out-occamy/vettori-paper.pt
Uso: lancia.sh paper"""
import json
import os
import urllib.request

import numpy as np
import torch
from safetensors import safe_open

D = "/data/memoria-episodica-affettiva/research/long-context/6-template-lens"
URL = os.environ.get("PONTE_URL", "http://127.0.0.1:8099")
LAYER = list(range(15, 33))
ALFA = 0.1
HF = "/data/jspace/hf/hub/models--Accio-Lab--occamy-1.0/snapshots/72ca96031d8d11650882b5fae8dad88412a76155"
LENTE = "/data/jspace/out-occamy/lente35b-18layer.pt"
OUT = "/data/jspace/out-occamy/vettori-paper.pt"

EMO = json.load(open("/data/memoria-episodica-affettiva/emozioni.json"))
ITALIANO = {x["en"]: x["it"] for x in EMO}   # nei log i nomi in italiano


def leggi(testo, layer):
    body = {"content": testo, "layer": layer, "per_token": True, "ultimi": 1}
    r = urllib.request.Request(URL + "/lux-read", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return np.asarray(json.load(urllib.request.urlopen(r, timeout=600))["states"][-1], np.float32)


def passaggi():
    out = []
    for x in EMO:
        righe = [r.strip() for r in open(f"{D}/passaggi/{x['it']}.txt", encoding="utf-8") if r.strip()]
        assert len(righe) == 30, (x["it"], len(righe))
        out += [(x["en"], r) for r in righe]
    return out


def avanzamento(letti, totale, ultimo):
    """12/10: la pagina /vettori/ legge questo file ogni 2 secondi (tempo reale, passaggio per passaggio)"""
    os.makedirs(f"{D}/vettori-pagina", exist_ok=True)
    tmp = f"{D}/vettori-pagina/avanzamento.json.tmp"
    json.dump({"letti": letti, "totale": totale, "emo": ITALIANO[ultimo[0]], "testo": ultimo[1]},
              open(tmp, "w", encoding="utf-8"), ensure_ascii=False)
    os.replace(tmp, f"{D}/vettori-pagina/avanzamento.json")


def stati(testi):
    f, ff = f"{D}/stati-template.npy", f"{D}/fatti.txt"
    S = np.load(f) if os.path.exists(f) else np.zeros((len(testi), len(LAYER), 2048), np.float32)
    fatti = int(open(ff).read()) if os.path.exists(ff) else 0
    for i in range(fatti, len(testi)):
        for j, L in enumerate(LAYER):
            S[i, j] = leggi(testi[i][1], L)
        avanzamento(i + 1, len(testi), testi[i])
        if (i + 1) % 30 == 0 or i + 1 == len(testi):
            np.save(f, S)
            open(ff, "w").write(str(i + 1))
            print(f"   {i + 1}/{len(testi)} passaggi ({ITALIANO[testi[i][0]]})", flush=True)
    return S


def template(S, etichette):
    nomi = [x["en"] for x in EMO]
    v = {e: {} for e in nomi}
    for j, L in enumerate(LAYER):
        X = S[:, j].astype(np.float64)
        mu = np.stack([X[etichette == e].mean(0) for e in nomi])
        R = X - mu[[nomi.index(e) for e in etichette]]
        Sw = R.T @ R / (len(X) - len(nomi))
        Sw = (1 - ALFA) * Sw + ALFA * np.trace(Sw) / Sw.shape[0] * np.eye(Sw.shape[0])
        inv = np.linalg.inv(Sw)
        for i, e in enumerate(nomi):
            c = mu[i] - np.delete(mu, i, 0).mean(0)
            d = inv @ c
            v[e][L] = torch.tensor(d / np.linalg.norm(d), dtype=torch.float32)
    return v


def jlens(tok_id):
    lente = torch.load(LENTE, weights_only=False, map_location="cpu")["J"]
    with safe_open(f"{HF}/model-00012-of-00013.safetensors", "pt") as f:
        g = 1.0 + f.get_tensor("model.language_model.norm.weight").float()
        W = f.get_slice("lm_head.weight")
        righe = {e: W[t:t + 1].float()[0] * g for e, t in tok_id.items()}
    v = {e: {} for e in tok_id}
    for L in LAYER:
        J = lente[L].float()
        for e, u in righe.items():
            d = J.T @ u
            v[e][L] = d / d.norm()
    return v


def main():
    import transformers
    tok = transformers.AutoTokenizer.from_pretrained(HF)
    un_token = {}
    for x in EMO:
        ids = tok.encode(" " + x["it"], add_special_tokens=False)
        if len(ids) == 1:
            un_token[x["en"]] = ids[0]
    print(f"parole di un token: {[x['it'] for x in EMO if x['en'] in un_token]}", flush=True)
    testi = passaggi()
    S = stati(testi)
    et = np.array([e for e, _ in testi])
    tl = template(S, et)
    jl = jlens(un_token)
    vettori = {e: (jl[e] if e in jl else tl[e]) for e in tl}
    metodo = {e: ("jlens" if e in jl else "template") for e in tl}
    for e in jl:   # quanto concordano i due metodi sulle parole di un token
        cos = [float(jl[e][L] @ tl[e][L]) for L in LAYER]
        print(f"   {ITALIANO[e]}: coseno J-lens/template L15 {cos[0]:.2f} L24 {cos[9]:.2f} L29 {cos[14]:.2f}", flush=True)
    torch.save({"vettori": vettori, "metodo": metodo, "template": tl, "banda": LAYER, "alfa": ALFA,
                "data": "2026-10-12", "fonte": "paper workspace: J-lens (un token) + template lens (passaggi/)",
                "n_passaggi": {e: int((et == e).sum()) for e in tl}}, OUT)
    print(f"salvato {OUT}", flush=True)


if __name__ == "__main__":
    main()
