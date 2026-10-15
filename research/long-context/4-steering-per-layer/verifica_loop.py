#!/usr/bin/env python3
"""13/10 sera: "il loop e' una dose eccessiva, assumiamo sia cosi' e verifichiamo".
Risposte intere al contesto 0 (come completi() di steering_layer.py) per le emozioni andate in loop, sullo stesso
layer a dosi crescenti dentro il loro range efficace: se il loop compare solo sopra una certa dose, l'ipotesi tiene.
Si lancia SOLO a prova ferma: il vettore sul server vale per tutte le richieste. Uscita: verifica-loop.jsonl"""
import json
import re
import sys
import time

sys.path.insert(0, "4-steering-per-layer")
import steering_layer as S

PROVE = {"optimism": (26, [0.10, 0.16, 0.24, 0.28]),
         "terror": (26, [0.68, 0.76, 0.84, 0.88]),
         "annoyance": (26, [0.162, 0.18, 0.25, 0.33])}


def in_loop(t):
    return bool(re.search(r"(.{8,80}?)\1{4,}", t[-3000:], re.S))


x = S.contesti()[0]
S.prepara(x, None)
with open(f"{S.D}/verifica-loop.jsonl", "a") as out:
    for e, (L, dosi) in PROVE.items():
        d = (S.IT[e][L].float() / S.IT[e][L].float().norm()).numpy()
        for dose in dosi:
            S.T.ponte._post("/control-vector", {"layers": {str(L): (dose * S.T.SCALA * d).tolist()}, "relative": True})
            t0 = time.time()
            try:
                testo, fine = S.genera_intero(x["testo"])
            finally:
                S.T.ponte.calma()
            out.write(json.dumps({"emo": e, "layer": L, "dose": dose, "loop": in_loop(testo), "caratteri": len(testo),
                                  "stop": fine, "testo": testo}, ensure_ascii=False) + "\n")
            out.flush()
            print(f"{e} L{L} {dose}: {'LOOP' if in_loop(testo) else 'ok'}, {len(testo)} caratteri, {fine}, "
                  f"{time.time() - t0:.0f}s", flush=True)
