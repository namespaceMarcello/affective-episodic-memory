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


# 15/10: il contesto deve avere TUTTO quello che l'agente riceve: prompt di sistema E definizioni dei tool (Hermes le
# manda a parte e il template di Occamy le mette in <tools>; prima mancavano). Tool veri: dati/tool-agente.json.
TOOLS = json.load(open(f"{DATI}/tool-agente.json"))["tools"]
PASSI, MAX_TOKEN = 10, 140000   # 15/10: 10 contesti, dal solo sistema+tool fino a 140k token


def rendi(ms):
    return tok.apply_chat_template(ms, tools=TOOLS, tokenize=False, add_generation_prompt=True)


def n(testo):
    return len(tok(testo, add_special_tokens=False).input_ids)


def contesti():
    """[{k, temi, n_token, prefisso, testo}]: 10 contesti a passi uguali da C0 (prompt di sistema + tool + messaggio
    finale) a ~140k token. La conversazione e' quella dei blocchi a tema (blocchi-temi.json) in fila; ogni contesto si
    taglia prima di un messaggio dell'utente, il piu' vicino al suo passo. Prefisso K = prefisso del K+1 (cache)."""
    sessioni = os.environ["SESSIONI"].split()   # come estrai_contesti.py
    sistema = c.execute("select system_prompt from sessions where id=?", (sessioni[-1],)).fetchone()[0]
    finale = [m for m in sum((messaggi(s) for s in sessioni), []) if m["role"] == "user" and len(m["content"]) > 40][-1]
    fila, tema_di = [], []
    for b in json.load(open(f"{DATI}/blocchi-temi.json")):
        ms = messaggi(b["session_id"], b["primo_id"], b["ultimo_id"])
        fila += ms
        tema_di += [b["tema"]] * len(ms)
    sis = [{"role": "system", "content": sistema}]
    tagli = [0] + [i for i, m in enumerate(fila) if m["role"] == "user" and i > 0] + [len(fila)]
    lung = {t: n(rendi(sis + fila[:t] + [finale])) for t in tagli}
    t0 = lung[0]
    out, presi = [], []
    for k in range(PASSI):
        meta = t0 + k * (MAX_TOKEN - t0) / (PASSI - 1)
        t = min((t for t in tagli if lung[t] <= MAX_TOKEN and t not in presi), key=lambda t: abs(lung[t] - meta))
        presi.append(t)
        testo = rendi(sis + fila[:t] + [finale])
        prefisso = testo[:testo.rindex("<|im_start|>user")]
        assert not out or prefisso.startswith(out[-1]["prefisso"]), f"il contesto {k} non estende il {k - 1}"
        out.append({"k": k, "temi": list(dict.fromkeys(tema_di[:t])), "n_token": lung[t], "messaggi": t,
                    "prefisso": prefisso, "testo": testo})
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
            elif r.get("fine_rifinitura"):
                s.add(("rif", r["k"], r["emo"], r["layer"]))
            elif r.get("completo"):
                s.add(("completo", r["k"], r["emo"]))
            elif r.get("libero"):
                s.add(("libero", r["k"], r["emo"]))
            elif r.get("fine_risposte"):
                s.add(("risposte", r["k"], r["emo"]))
            elif r.get("base_completo"):
                s.add(("base_completo", r["k"]))
    return s


# 15/10: OGNI punto e' la risposta vera dell'agente: vettore acceso fino al primo punto di fine frase o a </think>,
# poi spento; si continua fino a 256 token e ci si ferma prima al token di fine o a </tool_call>. Niente grammatica,
# niente rifinitura, niente giudice. Sul punto: le misure dello steering (sul ragionamento) e l'esame della tool call.
N_PUNTO = 256
FINE_TOOL = "</tool_call>"
ABBREVIAZIONI = {"es", "ecc", "etc", "cfr", "pag", "pagg", "eg", "ie", "vs", "dott", "sig", "sigg", "prof", "ing",
                 "avv", "nr", "num", "art", "cap", "fig", "vol", "ca", "mr", "mrs", "dr"}


def fine_frase(t):
    """14/10 (l'utente "fino al primo punto di fine frase"): "." finale che chiude una frase. Non valgono "dormi.sh"
    (lo spazio dopo lo controlla chi chiama), "...", il numero di un elenco ("1."), i decimali e le abbreviazioni."""
    m = re.search(r"(\S*)\.$", t)
    if not m or t.endswith(".."):
        return False
    parola = m.group(1).lower().replace(".", "")
    if re.fullmatch(r"[\d.,]*", parola):   # "1." "2.3." o un punto da solo
        return False
    return parola.strip("(\"'*_`[") not in ABBREVIAZIONI


def genera_punto(testo, vettore=None):
    """vettore = (layer, componenti) o None (base). Fase 1 col vettore, un token alla volta (temperatura 0): si ferma al
    primo punto di fine frase (il token dopo comincia con uno spazio: quel token si scarta e lo rigenera la fase 2), a
    </think> (idem) o al token di fine. Fase 2 senza vettore fino a N_PUNTO token in tutto, stop al token di fine o a
    </tool_call>. La cache del prefisso si riusa: le richieste estendono sempre lo stato precedente."""
    ids = tok(testo, add_special_tokens=False).input_ids
    gen, fine, n1 = [], None, 0
    if vettore is not None:
        T.ponte._post("/control-vector", {"layers": {str(vettore[0]): vettore[1].tolist()}, "relative": True})
    try:
        while len(gen) < N_PUNTO:
            r = post("/completion", {"prompt": ids + gen, "n_predict": 1, "temperature": 0, "cache_prompt": True,
                                     "return_tokens": True})
            pezzo = tok.decode(r["tokens"])
            if fine_frase(tok.decode(gen)) and pezzo[:1].isspace():
                break
            if "</think>" in pezzo:
                break
            if r.get("stop_type") == "eos" or not r["tokens"]:
                fine = "eos"
                break
            gen += r["tokens"]
    finally:
        T.ponte.calma()
    n1 = len(gen)
    if fine is None and len(gen) < N_PUNTO:
        r = post("/completion", {"prompt": ids + gen, "n_predict": N_PUNTO - len(gen), "temperature": 0,
                                 "cache_prompt": True, "stop": [FINE_TOOL]})
        coda = r["content"] + (FINE_TOOL if r.get("stop_type") == "word" else "")
        fine = {"word": "tool_call", "eos": "eos"}.get(r.get("stop_type"), "limite")
        return tok.decode(gen) + coda, fine, n1
    return tok.decode(gen), fine or "limite", n1


TOOL_CALL = re.compile(r"<tool_call>\s*<function=([^>\n]+)>\s*((?:<parameter=[^>\n]+>.*?</parameter>\s*)*)</function>\s*</tool_call>", re.S)
PARAMETRO = re.compile(r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", re.S)
SCHEMI = {t["function"]["name"]: t["function"].get("parameters", {}) for t in TOOLS}


def esame_tool(testo):
    """15/10: la tool call dopo il ragionamento nel formato di Occamy: presente, ben formata, strumento fra le
    definizioni vere (dati/tool-agente.json), parametri esistenti nello schema e obbligatori presenti."""
    dopo = testo.split("</think>", 1)[1] if "</think>" in testo else ""
    if "<tool_call>" not in dopo:
        return {"tool_presente": False, "tool_valida": False, "tool_errore": "nessuna tool call"}
    blocchi = re.findall(r"<tool_call>.*?(?:</tool_call>|$)", dopo, re.S)
    chiamate, errori = [], []
    for b in blocchi:
        m = TOOL_CALL.fullmatch(b.strip())
        if not m:
            errori.append("forma")
            continue
        nome = m.group(1).strip()
        par = {k.strip(): v for k, v in PARAMETRO.findall(m.group(2))}
        chiamate.append({"nome": nome, "parametri": sorted(par)})
        sc = SCHEMI.get(nome)
        if sc is None:
            errori.append(f"strumento inesistente {nome}")
            continue
        props = sc.get("properties", {})
        errori += [f"{nome}: parametro inesistente {k}" for k in par if k not in props]
        errori += [f"{nome}: manca {k}" for k in sc.get("required", []) if k not in par]
    return {"tool_presente": True, "tool_valida": not errori, "tool_errore": "; ".join(errori) or None,
            "tool": chiamate}


def ragionamento(testo):
    return testo.split("</think>", 1)[0]


def main():
    ctxs = contesti()
    json.dump([{k: v for k, v in x.items() if k not in ("prefisso", "testo")} for x in ctxs],
              open(f"{D}/contesti-layer.json", "w"), indent=1, ensure_ascii=False)
    for x in ctxs:
        print(f"contesto {x['k']}: {x['n_token']} token, temi {x['temi']}", flush=True)
    gia = fatti()
    out = open(OUT, "a")
    # 15/10: un'emozione alla volta su tutti i 10 contesti. La base senza vettore si fa UNA volta per contesto
    # (si rilegge dal jsonl alle riprese), con la stessa procedura dei punti.
    basi = {}
    for riga in open(OUT) if os.path.exists(OUT) else []:
        r = json.loads(riga)
        if r.get("base"):
            basi[r["k"]] = r["testo"]
    stati_base = {}
    for e in T.NOMI:
        t_e = time.time()
        for i, x in enumerate(ctxs):
            if all((x["k"], e, l) in gia for l in LAYER):
                continue
            print(f"== contesto {x['k']} ({x['n_token']} token) {e}", flush=True)
            T.ponte.calma()
            prepara(x, ctxs[i - 1] if i else None)
            if x["k"] not in basi:
                t0 = time.time()
                base, fine, n1 = genera_punto(x["testo"])
                basi[x["k"]] = base
                out.write(json.dumps({"base": True, "k": x["k"], "n_token": x["n_token"], "testo": base, "stop": fine,
                                      "token_col_vettore": n1, **esame_tool(base)}, ensure_ascii=False) + "\n")
                out.flush()
                print(f"   base ({time.time() - t0:.0f}s, {fine}): {base[:90]!r}", flush=True)
            if x["k"] not in stati_base:
                stati_base[x["k"]] = T.stato(ragionamento(basi[x["k"]]))
            base, sb = basi[x["k"]], stati_base[x["k"]]
            t0, npunti = time.time(), 0
            for layer in LAYER:
                if (x["k"], e, layer) in gia:
                    continue
                serie = []
                for dose in DOSI:
                    r = punto(x["testo"], base, sb, e, layer, dose)
                    npunti += 1
                    serie.append(r)
                    out.write(json.dumps(dict(r, k=x["k"], n_token=x["n_token"]), ensure_ascii=False) + "\n")
                    perche = basta(serie)
                    if perche:
                        break
                out.write(json.dumps({"fine": True, "k": x["k"], "emo": e, "layer": layer, "stop": perche,
                                      "ultima_dose": dose}) + "\n")
                out.flush()
                gia.add((x["k"], e, layer))
            print(f"   {e} C{x['k']}: {npunti} punti in {time.time() - t0:.0f}s", flush=True)
            import analizza_layer
            analizza_layer.main()
        print(f"== {e}: tutti i contesti in {time.time() - t_e:.0f}s", flush=True)

# 12/10: la parola intrusa si cerca in ITALIANO (nome italiano di emozioni.json), parola intera:
# singolare e plurale (gioia/gioie, colpa/colpe ma non colpo), invariata se finisce in -i o accentata
ITALIANO = {x["en"]: x["it"] for x in json.load(open("/data/memoria-episodica-affettiva/emozioni.json"))}


def parola_italiana(testo, e):
    p = ITALIANO[e].lower()
    flessione = {"a": "[ae]", "o": "[oi]", "e": "[ei]"}   # singolare/plurale; -i e accentate invariate
    forma = re.escape(p[:-1]) + flessione[p[-1]] if p[-1] in flessione else re.escape(p)
    return re.search(r"\b" + forma + r"\b", testo.lower()) is not None


RANGO_MAX = 4   # 15/10: efficace = faccia giusta fra le prime 4 delle 51 (prima 3)


def efficace(r):
    return r["cambiato"] and not r["parola"] and not r["rotto"] and (r.get("rango") or 99) <= RANGO_MAX


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


# 12/10 sera: vettori col metodo del paper workspace (6-template-lens/estrai_vettori.py): J-lens per le parole
# italiane di un token, template lens per le altre. Prima: contrastivi sul diario (emovec-it.pt, corsa in archivio/).
# "en" = i vecchi (gradiente sulla parola inglese / contrastivi sui ricordi), solo per il confronto.
IT = torch.load("/data/jspace/out-occamy/vettori-paper.pt", weights_only=False)["vettori"]


def punto(testo_ctx, base, sb, e, layer, dose, vettori="it"):
    """15/10: un layer, la risposta vera (genera_punto); misure dello steering sul ragionamento + esame della tool call"""
    fonte = IT[e] if vettori == "it" else T.ROSSI[e] if T.METODO[e] == "contrastivo" else T.GRAD[e]
    d = (fonte[layer].float() / fonte[layer].float().norm()).numpy()
    t, fine, n1 = genera_punto(testo_ctx, (layer, dose * T.SCALA * d))
    ra = ragionamento(t)
    r = {"emo": e, "layer": layer, "dose": dose, "cambiato": t != base, "parola": parola_italiana(t, e),
         "rotto": T.degenerato(ra) if ra.strip() else False, "testo": t, "stop": fine, "token_col_vettore": n1, **esame_tool(t)}
    if ra != ragionamento(base) and ra:
        dd = T.stato(ra) - sb
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
                        out.write(json.dumps(dict(r, k=x["k"], vettori=v, testo=r["testo"]), ensure_ascii=False) + "\n")
                        out.flush()
            print(f"   {e} fatto", flush=True)


if __name__ == "__main__":
    confronto() if sys.argv[1:] == ["confronto"] else main()
