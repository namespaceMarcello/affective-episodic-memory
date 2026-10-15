#!/usr/bin/env python3
"""14/10 (l'utente "Prova"): lo schema del 09/10, vettore acceso SOLO sul prefill del tratto in entrata e spento prima
della generazione. Il tratto qui e' il messaggio finale del contesto 0 (quello dell'utente su dormi.sh), senza frasi
nuove: il prefisso e' gia' nello slot senza vettore, il messaggio finale si elabora col vettore acceso (n_predict 1),
poi il vettore si spegne e la risposta riparte dalla cache, fino alla fine (tetto 2048).
Stesse 3 emozioni di verifica_think.py, dose migliore x1, x2, x4. Si lancia SOLO a prova ferma.
Uscita: verifica-prefill.jsonl"""
import json
import sys
import time

sys.path.insert(0, "4-steering-per-layer")
import steering_layer as S

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


PROVE = {"purification": (22, 0.06), "dominance": (24, 0.14), "amazement": (30, 0.95)}
FATTORI = (1, 2, 4)
if sys.argv[1:] == ["controllo"]:   # stesso percorso (token, n_predict 0, ripresa dalla cache), vettore a dose 0
    PROVE, FATTORI = {"purification": (22, 0.0)}, (1,)

x = S.contesti()[0]
IDS = S.tok(x["testo"], add_special_tokens=False).input_ids
# Occamy ha strati ricorrenti: la cache si riusa solo se lo stato e' ESATTAMENTE il prefisso della richiesta dopo.
# Col vettore si elabora tutto tranne l'ultimo token (n_predict 0); la risposta riparte da li' senza vettore.
with open(f"{S.D}/verifica-prefill.jsonl", "a") as out:
    S.prepara(x, None)
    S.T.ponte.calma()
    base, fine_b = S.genera_intero(x["testo"])
    out.write(json.dumps({"base": True, "tool_valida": tool_valida(base), "stop": fine_b, "testo": base},
                         ensure_ascii=False) + "\n")
    out.flush()
    print(f"base: tool valida {tool_valida(base)}, {fine_b}", flush=True)
    for e, (L, d0) in PROVE.items():
        d = (S.IT[e][L].float() / S.IT[e][L].float().norm()).numpy()
        for f in FATTORI:
            dose = round(d0 * f, 3)
            t0 = time.time()
            S.prepara(x, None)   # slot = prefisso senza vettore
            S.T.ponte._post("/control-vector", {"layers": {str(L): (dose * S.T.SCALA * d).tolist()}, "relative": True})
            try:
                r1 = S.post("/completion", {"prompt": IDS[:-1], "n_predict": 0, "temperature": 0, "cache_prompt": True})
            finally:
                S.T.ponte.calma()
            r2 = S.post("/completion", {"prompt": IDS, "n_predict": S.MAX_COMPLETO, "temperature": 0,
                                        "cache_prompt": True})
            testo = r2["content"]
            fine = r2.get("stop_type") or ("eos" if r2.get("stopped_eos") else "limite")
            riga = {"emo": e, "layer": L, "dose": dose, "token_col_vettore": r1["timings"]["prompt_n"],
                    "token_rielaborati": r2["timings"]["prompt_n"], "cambiato": testo != base,
                    "parola": S.parola_italiana(testo, e), "tool_valida": tool_valida(testo), "stop": fine,
                    "testo": testo}
            out.write(json.dumps(riga, ensure_ascii=False) + "\n")
            out.flush()
            print(f"{e} L{L} {dose}: {riga['token_col_vettore']} token col vettore, {riga['token_rielaborati']} "
                  f"rielaborati senza, cambiato {riga['cambiato']}, parola {riga['parola']}, tool valida "
                  f"{riga['tool_valida']}, {fine}, {time.time() - t0:.0f}s", flush=True)
