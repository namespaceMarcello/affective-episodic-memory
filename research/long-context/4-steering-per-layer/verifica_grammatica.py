#!/usr/bin/env python3
"""14/10: al primo punto singolo del ragionamento si mette la fine del ragionamento, poi la risposta e'
vincolata dalla GRAMMATICA della tool call di Occamy (formato del suo chat template:
<tool_call>\\n<function=NOME>\\n<parameter=NOME>\\nvalore\\n</parameter>\\n</function>\\n</tool_call>).
Ragionamento un token alla volta (n_predict 1, prompt a token): la cache e' sempre esattamente il prefisso della
richiesta dopo, cosi' il vettore resta nella cache. Il vettore e' acceso solo sul ragionamento e si spegne prima
della tool call. Senza vettore e con le 3 emozioni di verifica_think.py. Si lancia SOLO a prova ferma.
Uscita: verifica-grammatica.jsonl"""
import json
import re
import sys
import time

sys.path.insert(0, "4-steering-per-layer")
import steering_layer as S

PROVE = {"senza vettore": (None, 0.0), "purification": (22, 0.06), "dominance": (24, 0.14), "amazement": (30, 0.95)}
MAX_RAGIONAMENTO = 300
CHIUSURA = "\n</think>\n\n"
GRAMMATICA = r'''root ::= "<tool_call>\n<function=" nome ">\n" parametro+ "</function>\n</tool_call>"
nome ::= [a-zA-Z0-9_.-]+
parametro ::= "<parameter=" nome ">\n" valore "\n</parameter>\n"
valore ::= [^<]+
'''


def ids(t):
    return S.tok(t, add_special_tokens=False).input_ids


x = S.contesti()[0]
IDS = ids(x["testo"])
with open(f"{S.D}/verifica-grammatica.jsonl", "a") as out:
    for e, (L, dose) in PROVE.items():
        t0 = time.time()
        S.prepara(x, None)
        S.T.ponte.calma()
        if L is not None:
            d = (S.IT[e][L].float() / S.IT[e][L].float().norm()).numpy()
            S.T.ponte._post("/control-vector", {"layers": {str(L): (dose * S.T.SCALA * d).tolist()}, "relative": True})
        gen, perche = [], "tetto"
        try:
            for _ in range(MAX_RAGIONAMENTO):
                r = S.post("/completion", {"prompt": IDS + gen, "n_predict": 1, "temperature": 0, "cache_prompt": True,
                                           "return_tokens": True})
                nuovo = r["tokens"]
                testo = S.tok.decode(gen)
                pezzo = S.tok.decode(nuovo)
                # punto singolo = "." non preceduto da un altro "." e seguito da spazio o a capo (non "dormi.sh", non "...")
                if re.search(r"(?<!\.)\.$", testo) and (pezzo[:1].isspace()):
                    perche = "punto"
                    break   # il token nuovo non e' ancora nella cache: si scarta
                if "</think>" in pezzo or r.get("stop_type") == "eos":
                    perche = "fine ragionamento da sola"
                    break
                gen += nuovo
        finally:
            S.T.ponte.calma()
        ragionamento = S.tok.decode(gen)
        r2 = S.post("/completion", {"prompt": IDS + gen + ids(CHIUSURA), "n_predict": 512, "temperature": 0,
                                    "cache_prompt": True, "grammar": GRAMMATICA})
        chiamata = r2["content"]
        m = re.match(r"<tool_call>\n<function=([^>]+)>\n(.*)</function>\n</tool_call>$", chiamata, re.S)
        riga = {"emo": e, "layer": L, "dose": dose, "token_ragionamento": len(gen), "perche": perche,
                "ragionamento": ragionamento, "chiamata": chiamata, "completa": bool(m),
                "funzione": m.group(1) if m else None, "token_rielaborati": r2["timings"]["prompt_n"],
                "stop": r2.get("stop_type")}
        out.write(json.dumps(riga, ensure_ascii=False) + "\n")
        out.flush()
        print(f"{e}: {len(gen)} token di ragionamento ({perche}), chiamata completa {riga['completa']}, "
              f"funzione {riga['funzione']}, {time.time() - t0:.0f}s", flush=True)
