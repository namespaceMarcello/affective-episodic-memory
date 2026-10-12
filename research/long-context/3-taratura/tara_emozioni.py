#!/usr/bin/env python3
"""Taratura dei vettori emotivi SUL SERVER (12/10/2026): dose e finestra di layer per ognuna delle 51 emozioni,
misurate su Occamy servito da luxifer con il contesto vero "sys" (prompt di sistema dell'agente + messaggio dell'utente),
dopo la correzione di scala in ponte.dirs_emozione (la taratura del 28/07 era in HF con prompt corti).
Per ogni emozione: finestre candidate {15, 21, 26} + quella di emo-cvec-v2, dosi 0.025..0.8 (unita' HF:
h += dose*||h||*dir, al server si manda dose*sqrt(2048)*dir in modo relativo). Vettori: emovec-18 (gradiente) o
emovec-rossi (contrastivi, le 13 del gruppo rosso), come assembla-emocvec.py.
Misure per punto: testo diverso dalla base, eco = coseno fra lo spostamento a L29-lente del testo generato (sterzato
meno base, letto da solo, vettore spento) e la faccia vera dell'emozione (media dei ricordi dell'agente taggati con lei),
rango della faccia giusta fra le 51, parola dell'emozione intrusa, testo rotto (criteri di scan-c2).
Scelta: fra i punti puliti (testo cambiato, niente intrusione, non rotto) quello con eco piu' alta -> finestra e
centro; bordo = prima dose della finestra scelta con intrusione o testo rotto (0.8 se nessuna).
Verifica: le scelte a 32k e 140k (stesse misure). Uscite: taratura-emozioni.json, emo-cvec-v3.npz in out-occamy
(stesso formato di emo-cvec-v2; lo scambio in produzione e' un passo separato, con backup)."""
import json
import os
import sqlite3
import sys
import urllib.request

import numpy as np
import torch

sys.path.insert(0, "/data/memoria-episodica-affettiva")
os.environ.setdefault("PONTE_URL", "http://127.0.0.1:8099")
import ponte  # noqa: E402

URL = os.environ["PONTE_URL"]
D = "/data/memoria-episodica-affettiva/research/long-context/3-taratura"
DATI = "/data/memoria-episodica-affettiva/research/long-context/dati"
OUT35 = "/data/jspace/out-occamy"
FINESTRE = [15, 21, 26]
DOSI = [0.025, 0.05, 0.1, 0.2, 0.4, 0.8]
N = 64

V2 = np.load("/data/memoria-episodica-affettiva/emo-cvec-v2.npz")
NOMI = [str(n) for n in V2["nomi"]]
PAROLE = dict(zip(NOMI, (str(p) for p in V2["parola"])))
METODO = dict(zip(NOMI, (str(m) for m in V2["metodo"])))
FIN_V2 = dict(zip(NOMI, (int(l[0]) for l in V2["layers"])))
GRAD = torch.load(f"{OUT35}/emovec-18.pt", weights_only=True)["vectors"]
ROSSI = torch.load(f"{OUT35}/emovec-rossi.pt", weights_only=True)["vettori"]
SCALA = np.sqrt(V2["dirs"].shape[-1])


def direzioni(e, w):
    fonte = ROSSI[e] if METODO[e] == "contrastivo" else GRAD[e]
    return [(fonte[l].float() / fonte[l].float().norm()).numpy() for l in (w, w + 1, w + 2)]


def facce():
    z = np.load(f"{OUT35}/lux-L29.npz")
    pos = {int(i): k for k, i in enumerate(z["ids"])}
    db = sqlite3.connect("file:/data/workspace/memoria/memoria.db?mode=ro", uri=True)
    tag = dict(db.execute("SELECT id, emo_tag FROM nodi WHERE classe='vissuto'"))
    db.close()
    F, n = [], {}
    for e in NOMI:
        idx = [pos[i] for i, t in tag.items() if t == e and i in pos]
        n[e] = len(idx)
        f = (z["stati"][idx] - z["mu"]).mean(0)
        F.append(f / np.linalg.norm(f))
    return np.stack(F), n


FACCE, N_RICORDI = facce()


def completa(testo):
    body = {"prompt": testo, "n_predict": N, "temperature": 0, "cache_prompt": True}
    r = urllib.request.Request(URL + "/completion", json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        return json.load(urllib.request.urlopen(r, timeout=7200))["content"]
    except urllib.error.HTTPError as err:   # 500 "does not match the expected Content-only format": uscita rotta
        if err.code != 500:
            raise
        return ""


def stato(testo):
    return np.asarray(ponte.leggi_sonda(testo)["mean"])


def intrusione(testo, parola):
    return parola.lower()[:max(4, len(parola) - 2)] in testo.lower()


def degenerato(testo):
    if len(testo.strip()) < 10:
        return True
    parole = testo.lower().split()
    if len(parole) >= 8 and len(set(parole)) / len(parole) < 0.4:
        return True
    return sum(1 for c in testo if ord(c) > 0x2000) / max(len(testo), 1) > 0.15


def punto(testo_ctx, base, sb, e, w, dose):
    dirs = direzioni(e, w)
    ponte._post("/control-vector", {"layers": {str(l): (dose * SCALA * d).tolist() for l, d in zip((w, w + 1, w + 2), dirs)},
                                    "relative": True})
    try:
        t = completa(testo_ctx)
    finally:
        ponte.calma()
    r = {"emo": e, "finestra": w, "dose": dose, "cambiato": t != base,
         "parola": intrusione(t, PAROLE[e]) or intrusione(t, e), "rotto": degenerato(t), "testo": t}
    if r["cambiato"] and t:   # testo vuoto (EOS subito): /lux-read risponde 400
        d = stato(t) - sb
        c = FACCE @ (d / (np.linalg.norm(d) + 1e-9))
        r["eco"] = float(c[NOMI.index(e)])
        r["rango"] = int((c > c[NOMI.index(e)]).sum()) + 1
    return r


def scrivi(stato_json):
    json.dump(stato_json, open(f"{D}/taratura-emozioni.json", "w"), indent=1, ensure_ascii=False)


contesti = {c["nome"]: c for c in json.load(open(f"{DATI}/contesti.json"))}
ris = {"n_ricordi": N_RICORDI, "punti": [], "scelte": {}, "verifica": []}
if os.environ.get("RIPRENDI"):   # riparte dalle emozioni gia' scelte in taratura-emozioni.json
    ris = json.load(open(f"{D}/taratura-emozioni.json"))
    ris["punti"] = [r for r in ris["punti"] if r["emo"] in ris["scelte"]]

# --- 1. scansione sul contesto sys
ctx = contesti["sys"]["testo"]
ponte.calma()
base = completa(ctx)
sb = stato(base)
print(f"== sys base: {base[:90]!r}", flush=True)
for e in NOMI:
    if e in ris["scelte"]:
        continue
    for w in sorted(set(FINESTRE) | {FIN_V2[e]}):
        for dose in DOSI:
            r = punto(ctx, base, sb, e, w, dose)
            ris["punti"].append(r)
            print(f"   {e} L{w} {dose}: {'cambiato' if r['cambiato'] else 'uguale'} eco {r.get('eco', 0):+.2f} "
                  f"rango {r.get('rango', '-')}{' PAROLA' if r['parola'] else ''}{' ROTTO' if r['rotto'] else ''}", flush=True)
            if r["rotto"]:
                break
    mie = [r for r in ris["punti"] if r["emo"] == e]
    pulite = [r for r in mie if r["cambiato"] and not r["parola"] and not r["rotto"]]
    if pulite:
        b = max(pulite, key=lambda r: r["eco"])
        w, centro = b["finestra"], b["dose"]
    else:   # nessun punto pulito: resta la finestra di v2, dose minima
        b, w, centro = None, FIN_V2[e], DOSI[0]
    bordo = next((r["dose"] for r in mie if r["finestra"] == w and r["dose"] > centro and (r["parola"] or r["rotto"])), 0.8)
    ris["scelte"][e] = {"finestra": w, "centro": centro, "bordo": bordo, "eco": b["eco"] if b else None,
                        "rango": b["rango"] if b else None, "pulito": b is not None}
    print(f"-> {e}: L{w}-{w + 2} centro {centro} bordo {bordo} eco {ris['scelte'][e]['eco']} "
          f"rango {ris['scelte'][e]['rango']}", flush=True)
    scrivi(ris)

# --- 2. artefatto v3 (stesso formato di v2)
dest = f"{OUT35}/emo-cvec-v3.npz"
np.savez(dest, nomi=np.array(NOMI),
         layers=np.array([[ris["scelte"][e]["finestra"] + k for k in range(3)] for e in NOMI], np.int64),
         dirs=np.stack([np.stack(direzioni(e, ris["scelte"][e]["finestra"])) for e in NOMI]).astype(np.float32),
         centro=np.array([ris["scelte"][e]["centro"] for e in NOMI], np.float32),
         bordo=np.array([ris["scelte"][e]["bordo"] for e in NOMI], np.float32),
         metodo=V2["metodo"], parola=V2["parola"],
         meta=json.dumps({"data": "2026-10-12", "sonda": 29, "modello": "Accio-Lab/occamy-1.0",
                          "fonti": ["emovec-18.pt", "emovec-rossi.pt", "taratura-emozioni.json"],
                          "taratura": "server luxifer, contesto sys, eco sul testo generato", "modo_server": "relative"}))
print(f"scritto {dest}", flush=True)

# --- 3. verifica delle scelte a 32k e 140k
for nome in ("32k", "140k"):
    ctx = contesti[nome]["testo"]
    ponte.calma()
    base = completa(ctx)
    sb = stato(base)
    print(f"== verifica {nome} base: {base[:90]!r}", flush=True)
    for e in NOMI:
        s = ris["scelte"][e]
        r = punto(ctx, base, sb, e, s["finestra"], s["centro"])
        r["contesto"] = nome
        ris["verifica"].append(r)
        print(f"   {e}: {'cambiato' if r['cambiato'] else 'uguale'} eco {r.get('eco', 0):+.2f} rango {r.get('rango', '-')}"
              f"{' PAROLA' if r['parola'] else ''}{' ROTTO' if r['rotto'] else ''}", flush=True)
        scrivi(ris)

print("\n| contesto | cambiati | eco media | rango 1 | rango <=5 | parola | rotti |")
print("|---|---|---|---|---|---|---|")
righe = [("sys", [dict(r, contesto="sys") for r in ris["punti"]
                  if (r["finestra"], r["dose"]) == (ris["scelte"][r["emo"]]["finestra"], ris["scelte"][r["emo"]]["centro"])])]
righe += [(n, [r for r in ris["verifica"] if r["contesto"] == n]) for n in ("32k", "140k")]
for nome, rr in righe:
    cam = [r for r in rr if r["cambiato"]]
    print(f"| {nome} | {len(cam)}/{len(rr)} | {np.mean([r['eco'] for r in cam]) if cam else float('nan'):+.3f} | "
          f"{sum(r.get('rango') == 1 for r in cam)} | {sum((r.get('rango') or 99) <= 5 for r in cam)} | "
          f"{sum(r['parola'] for r in rr)} | {sum(r['rotto'] for r in rr)} |")
