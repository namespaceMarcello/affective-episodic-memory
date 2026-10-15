#!/usr/bin/env python3
"""14/10 (l'utente, impostata da un'altra sessione): il vettore si SPEGNE al comparire di </think> oppure a 70 token
(quello che arriva prima), poi la risposta continua senza vettore fino alla fine (tetto 2048 in tutto). Contesto 0,
3 emozioni, una per tipo di guasto, layer e dose migliori da emozioni-layer/dati.json. Si guarda se la risposta chiude
con una tool call valida come la base. Si lancia SOLO a prova ferma: il vettore sul server vale per tutte le richieste.
Uscita: verifica-think.jsonl"""
import json
import sys
import time

sys.path.insert(0, "4-steering-per-layer")
import steering_layer as S

PROVE = {"purification": (22, 0.06),   # eco del messaggio OUT-OF-BAND + tool inventato
         "dominance": (24, 0.14),      # sintassi della tool call rotta
         "amazement": (30, 0.95)}      # non chiude mai </think>: prova il tetto dei 70
TETTO = 70
if sys.argv[1:] == ["controllo"]:   # 14/10 (l'utente "Fai"): stesso taglio a 70 token / </think>, SENZA vettore
    PROVE = {"controllo": (None, 0.0)}


def tool_valida(t):
    """come la base: in coda un JSON [{"name": ..., "arguments": {...}}]"""
    i = t.rfind("[{")
    if i < 0:
        return False
    try:
        x = json.loads(t[i:].strip())
    except ValueError:
        return False
    return isinstance(x, list) and len(x) > 0 and all(isinstance(c, dict) and "name" in c and "arguments" in c for c in x)


x = S.contesti()[0]
S.prepara(x, None)
with open(f"{S.D}/verifica-think.jsonl", "a") as out:
    S.T.ponte.calma()
    base, fine_b = S.genera_intero(x["testo"])
    out.write(json.dumps({"base": True, "tool_valida": tool_valida(base), "stop": fine_b, "testo": base},
                         ensure_ascii=False) + "\n")
    out.flush()
    print(f"base: tool valida {tool_valida(base)}, {fine_b}", flush=True)
    for e, (L, dose) in PROVE.items():
        t0 = time.time()
        if L is not None:
            d = (S.IT[e][L].float() / S.IT[e][L].float().norm()).numpy()
            S.T.ponte._post("/control-vector", {"layers": {str(L): (dose * S.T.SCALA * d).tolist()}, "relative": True})
        try:
            r1 = S.post("/completion", {"prompt": x["testo"], "n_predict": TETTO, "temperature": 0,
                                        "cache_prompt": True, "stop": ["</think>"]})
        finally:
            S.T.ponte.calma()
        t1, n1 = r1["content"], r1.get("tokens_predicted", 0)
        spento = "</think>" if r1.get("stop_type") == "word" else "70 token" if n1 >= TETTO else r1.get("stop_type")
        if spento == "</think>":
            t1 += "</think>"   # llama.cpp toglie la parola di stop dal content
        if spento in ("</think>", "70 token"):
            r2 = S.post("/completion", {"prompt": x["testo"] + t1, "n_predict": S.MAX_COMPLETO - n1,
                                        "temperature": 0, "cache_prompt": True})
            t2, fine = r2["content"], r2.get("stop_type") or ("eos" if r2.get("stopped_eos") else "limite")
        else:
            t2, fine = "", spento   # finita gia' col vettore acceso
        testo = t1 + t2
        out.write(json.dumps({"emo": e, "layer": L, "dose": dose, "token_fase1": n1, "spento": spento,
                              "tool_valida": tool_valida(testo), "stop": fine, "testo": testo,
                              "fase1": t1}, ensure_ascii=False) + "\n")
        out.flush()
        print(f"{e} L{L} {dose}: spento a {spento} ({n1} token), tool valida {tool_valida(testo)}, {fine}, "
              f"{len(testo)} caratteri, {time.time() - t0:.0f}s", flush=True)
