#!/usr/bin/env python3
"""Steering per layer (12/10/2026): efficacia e intensita' efficace di ognuna delle 51 emozioni su OGNI layer
che ha un vettore (L15..L32, un layer alla volta), dosi da 0.02 a passi di 0.02 (fino a 1.0, stop al primo testo
rotto), su 12 contesti di lunghezza crescente fino a ~140k token.
Contesti: prompt di sistema vero dell'agente + 0..11 blocchi da ~12k token di conversazioni vere, ognuno su un tema
diverso (blocchi-temi.json), + sempre lo stesso messaggio finale dell'utente (quello di contesti.json "sys").
Nessuna stringa nuova: solo testi gia' visti dall'agente.
Cache: per ogni contesto lo stato del server SENZA il messaggio finale si salva su disco (/slots save, prefisso-K.bin):
il contesto K+1 riparte dal K ed elabora solo il blocco nuovo; alla ripresa dopo un crash si ripristina da li'.
Misure per punto come tara_emozioni.py (testo cambiato, eco a L29-lente, rango della faccia giusta fra le 51,
parola intrusa, testo rotto). Uscita: steering-layer.jsonl (una riga per punto, si riprende da dove era).
Uso: lancia.sh layer"""
import json
import os
import re
import sqlite3
import sys
import time
import urllib.request

import numpy as np
import torch

os.environ.setdefault("PONTE_URL", "http://127.0.0.1:8099")
import types  # noqa: E402

import transformers  # noqa: E402

# di tara_emozioni.py servono solo le definizioni (vettori, facce, misure): si esegue fino all'inizio della taratura
_src = open("/data/memoria-episodica-affettiva/research/long-context/3-taratura/tara_emozioni.py").read()
_ns = {"__name__": "tara_emozioni"}
exec(compile(_src[:_src.index("contesti = {c[")], "tara_emozioni.py", "exec"), _ns)
T = types.SimpleNamespace(**_ns)

D = "/data/memoria-episodica-affettiva/research/long-context/4-steering-per-layer"
DATI = "/data/memoria-episodica-affettiva/research/long-context/dati"
URL = T.URL
DB = f"{DATI}/state.db"
OUT = f"{D}/steering-layer.jsonl"
LAYER = list(range(15, 33))
DOSI = [round(0.02 * k, 2) for k in range(1, 51)]
N_CTX = 143360

tok = transformers.AutoTokenizer.from_pretrained("Accio-Lab/occamy-1.0")
c = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)


def messaggi(sid, primo=None, ultimo=None):
    out = []
    for i, role, content, tool_calls, call_id in c.execute(
            "select id, role, content, tool_calls, tool_call_id from messages where session_id=? order by id", (sid,)):
        if (primo and i < primo) or (ultimo and i > ultimo):
            continue
        if role == "assistant":
            m = {"role": "assistant", "content": content or ""}
            if tool_calls:
                m["tool_calls"] = [{"type": "function", "function": {"name": t["function"]["name"],
                                    "arguments": json.loads(t["function"]["arguments"] or "{}")}}
                                   for t in json.loads(tool_calls)]
            out.append(m)
        elif role == "tool":
            out.append({"role": "tool", "content": content or "", "tool_call_id": call_id})
        elif role == "user" and (content or "").strip():
            out.append({"role": "user", "content": content})
    return out


def rendi(ms):
    return tok.apply_chat_template(ms, tokenize=False, add_generation_prompt=True)


def n(testo):
    return len(tok(testo, add_special_tokens=False).input_ids)


def contesti():
    """[{k, temi, n_token, prefisso, testo}]: testo = prefisso + messaggio finale; prefisso K e' prefisso del K+1"""
    sessioni = os.environ["SESSIONI"].split()   # come estrai_contesti.py
    sistema = c.execute("select system_prompt from sessions where id=?", (sessioni[-1],)).fetchone()[0]
    finale = [m for m in sum((messaggi(s) for s in sessioni), []) if m["role"] == "user" and len(m["content"]) > 40][-1]
    blocchi = json.load(open(f"{DATI}/blocchi-temi.json"))
    base = [{"role": "system", "content": sistema}]
    out, temi = [], []
    for k in range(len(blocchi) + 1):
        if k:
            b = blocchi[k - 1]
            base += messaggi(b["session_id"], b["primo_id"], b["ultimo_id"])
            temi.append(b["tema"])
        testo = rendi(base + [finale])
        coda = testo.rindex("<|im_start|>user")
        prefisso = testo[:coda]
        assert not out or prefisso.startswith(out[-1]["prefisso"]), f"il contesto {k} non estende il {k - 1}"
        nt = n(testo)
        if nt > N_CTX - T.N - 64:
            print(f"contesto {k}: {nt} token, oltre {N_CTX}: mi fermo a {k - 1}", flush=True)
            break
        out.append({"k": k, "temi": list(temi), "n_token": nt, "prefisso": prefisso, "testo": testo})
    return out


def post(path, body, timeout=7200):
    r = urllib.request.Request(URL + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=timeout))


def prepara(ctx, precedente):
    """porta lo slot allo stato 'prefisso del contesto' (dal file se c'e', altrimenti dal precedente + blocco nuovo)"""
    f = f"prefisso-{ctx['k']}.bin"
    if os.path.exists(f"{D}/slot/{f}"):
        post("/slots/0?action=restore", {"filename": f})
        return
    if precedente is not None:
        post("/slots/0?action=restore", {"filename": f"prefisso-{precedente['k']}.bin"})
    t0 = time.time()
    r = post("/completion", {"prompt": ctx["prefisso"], "n_predict": 1, "temperature": 0, "cache_prompt": True})
    print(f"   prefisso {ctx['k']}: {r['timings']['prompt_n']} token elaborati in {time.time() - t0:.0f}s", flush=True)
    post("/slots/0?action=save", {"filename": f})


def fatti():
    s = set()
    if os.path.exists(OUT):
        for riga in open(OUT):
            r = json.loads(riga)
            if r.get("fine"):
                s.add((r["k"], r["emo"], r["layer"]))
    return s


def main():
    ctxs = contesti()
    json.dump([{k: v for k, v in x.items() if k not in ("prefisso", "testo")} for x in ctxs],
              open(f"{D}/contesti-layer.json", "w"), indent=1, ensure_ascii=False)
    for x in ctxs:
        print(f"contesto {x['k']}: {x['n_token']} token, temi {x['temi']}", flush=True)
    gia = fatti()
    out = open(OUT, "a")
    prec = None
    for x in ctxs:
        if all((x["k"], e, l) in gia for e in T.NOMI for l in LAYER):
            prec = x
            continue
        print(f"== contesto {x['k']} ({x['n_token']} token)", flush=True)
        T.ponte.calma()
        prepara(x, prec)
        prec = x
        base = T.completa(x["testo"])
        sb = T.stato(base)
        print(f"   base: {base[:90]!r}", flush=True)
        out.write(json.dumps({"base": True, "k": x["k"], "n_token": x["n_token"], "testo": base}, ensure_ascii=False) + "\n")
        out.flush()
        for e in T.NOMI:
            t0, npunti = time.time(), 0
            for layer in LAYER:
                if (x["k"], e, layer) in gia:
                    continue
                serie = []
                for dose in DOSI:
                    r = punto(x["testo"], base, sb, e, layer, dose)
                    npunti += 1
                    serie.append(r)
                    out.write(json.dumps(dict(r, k=x["k"], n_token=x["n_token"], testo=r["testo"][:300]),
                                         ensure_ascii=False) + "\n")
                    perche = basta(serie)
                    if perche:
                        break
                out.write(json.dumps({"fine": True, "k": x["k"], "emo": e, "layer": layer, "stop": perche,
                                      "ultima_dose": dose}) + "\n")
                out.flush()
            print(f"   {e}: {npunti} punti in {time.time() - t0:.0f}s", flush=True)


# 12/10: la parola intrusa si cerca in ITALIANO (nome italiano di emozioni.json), parola intera:
# singolare e plurale (gioia/gioie, colpa/colpe ma non colpo), invariata se finisce in -i o accentata
ITALIANO = {x["en"]: x["it"] for x in json.load(open("/data/memoria-episodica-affettiva/emozioni.json"))}


def parola_italiana(testo, e):
    p = ITALIANO[e].lower()
    flessione = {"a": "[ae]", "o": "[oi]", "e": "[ei]"}   # singolare/plurale; -i e accentate invariate
    forma = re.escape(p[:-1]) + flessione[p[-1]] if p[-1] in flessione else re.escape(p)
    return re.search(r"\b" + forma + r"\b", testo.lower()) is not None


def efficace(r):
    return r["cambiato"] and not r["parola"] and not r["rotto"] and (r.get("rango") or 99) <= 3


def basta(serie):
    """12/10: appena si vede oversteering o calo di efficacia si passa al layer dopo, senza tutta la scala.
    Oversteering = testo rotto o parola dell'emozione nel testo. Calo = dopo almeno un punto efficace, 3 dosi di fila
    non efficaci; oppure 6 dosi di fila senza un'eco migliore della massima e l'ultima sotto la massima di 0.10."""
    r = serie[-1]
    if r["rotto"]:
        return "rotto"
    if r["parola"]:
        return "parola"
    if any(efficace(x) for x in serie) and len(serie) >= 3 and not any(efficace(x) for x in serie[-3:]):
        return "calo-efficacia"
    eco = [x.get("eco", 0.0) for x in serie]
    i = max(range(len(eco)), key=eco.__getitem__)
    if len(eco) - 1 - i >= 6 and eco[-1] < eco[i] - 0.10:
        return "calo-eco"
    return None


# 12/10, decisione dell'autrice: vettori in ITALIANO, contrastivi sul dataset diaristico (5-vettori-italiani/emovec_it.py).
# "en" = i vecchi (gradiente sulla parola inglese / contrastivi sui ricordi), solo per il confronto.
IT = torch.load("/data/jspace/out-occamy/emovec-it.pt", weights_only=False)["vettori"]


def punto(testo_ctx, base, sb, e, layer, dose, vettori="it"):
    """come T.punto ma su UN layer"""
    fonte = IT[e] if vettori == "it" else T.ROSSI[e] if T.METODO[e] == "contrastivo" else T.GRAD[e]
    d = (fonte[layer].float() / fonte[layer].float().norm()).numpy()
    T.ponte._post("/control-vector", {"layers": {str(layer): (dose * T.SCALA * d).tolist()}, "relative": True})
    try:
        t = T.completa(testo_ctx)
    finally:
        T.ponte.calma()
    r = {"emo": e, "layer": layer, "dose": dose, "cambiato": t != base,
         "parola": parola_italiana(t, e), "rotto": T.degenerato(t), "testo": t}
    if r["cambiato"] and t:
        dd = T.stato(t) - sb
        cc = T.FACCE @ (dd / (np.linalg.norm(dd) + 1e-9))
        r["eco"] = float(cc[T.NOMI.index(e)])
        r["rango"] = int((cc > cc[T.NOMI.index(e)]).sum()) + 1
    return r


def confronto():
    """12/10, decisione dell'autrice: stesso contesto, stessa emozione, stesso layer e dose, vettore italiano contro inglese"""
    EMO = ["joy", "sadness", "anger", "fear", "trust", "disgust", "surprise", "serenity"]
    ctxs = contesti()
    out = open(f"{D}/confronto-it-en.jsonl", "a")
    prec = None
    for x in ctxs:
        if x["k"] not in (0, ctxs[-1]["k"]):   # i prefissi intermedi servono a costruire l'ultimo
            prepara(x, prec)
            prec = x
            continue
        print(f"== confronto, contesto {x['k']} ({x['n_token']} token)", flush=True)
        T.ponte.calma()
        prepara(x, prec)
        prec = x
        base = T.completa(x["testo"])
        sb = T.stato(base)
        out.write(json.dumps({"base": True, "k": x["k"], "testo": base}, ensure_ascii=False) + "\n")
        for e in EMO:
            for layer in (18, 24, 29):
                for dose in (0.1, 0.2, 0.3, 0.4):
                    for v in ("it", "en"):
                        r = punto(x["testo"], base, sb, e, layer, dose, v)
                        out.write(json.dumps(dict(r, k=x["k"], vettori=v, testo=r["testo"][:300]), ensure_ascii=False) + "\n")
                        out.flush()
            print(f"   {e} fatto", flush=True)


if __name__ == "__main__":
    confronto() if sys.argv[1:] == ["confronto"] else main()
