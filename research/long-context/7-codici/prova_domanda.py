"""prova dei codici, seconda parte (16/10/2026): dopo la consegna del ricordo l'utente chiede all'agente il codice e il
contenuto. Frase dell'utente: DOMANDA. Come in produzione, se la domanda fa affiorare un ricordo, il riflesso lo consegna
subito dopo la domanda (richiamo vero del Lux, sola lettura).
  subito   conversazione + "ciao, come stai oggi?" + consegna + risposta vera dell'agente (campione 0 di prova_codici.py,
           solo se e' testo senza tool) + DOMANDA
  dopo10   consegna, almeno 10 messaggi veri, poi DOMANDA al posto del messaggio dell'utente seguente
Uscita: risultati-domanda.jsonl (codici trovati esatti/alterati/inventati e risposta intera, da leggere per il contenuto)"""
import json
import os
import time

import prova_codici as P

DOMANDA = "Dammi il codice del ricordo e descrivi per sommi capi il contenuto del ricordo."   # l'utente 16/10
OUT = f"{P.D}/risultati-domanda.jsonl"


def risposta_vera(k):
    for r in map(json.loads, open(P.OUT)):
        if r["k"] == k and r["caso"] == "subito" and r["campione"] == 0:
            testo = r["risposta"].split("</think>")[-1].strip()
            return None if "<tool_call>" in testo else testo
    return None


def main():
    ctxs = P.SL.contesti()
    sis, fila = P.fila_e_sistema()
    fatti = {(r["k"], r["caso"], r["campione"]) for r in map(json.loads, open(OUT))} if os.path.exists(OUT) else set()
    seconda = P.richiama([DOMANDA])[0]   # il ricordo che la domanda stessa farebbe affiorare
    for c in P.casi(ctxs, sis, fila):
        ms = list(c["ms"])
        if c["caso"] == "subito":
            vera = risposta_vera(c["k"])
            if vera is None:
                print(f"== C{c['k']} subito: la risposta vera dell'agente e' una chiamata a un tool, caso saltato", flush=True)
                continue
            ms.append({"role": "assistant", "content": vera})
        else:
            ms = ms[:-1]   # al posto del messaggio dell'utente seguente
        ms.append({"role": "user", "content": DOMANDA})
        consegne = list(c["consegne"])
        if seconda:
            ms += P.turni(seconda)
            consegne.append(seconda)
        prompt = P.SL.tok.apply_chat_template(ms, tools=P.TOOLS, tokenize=False, add_generation_prompt=True)
        codici = [v["codice"] for v in consegne]
        print(f"== C{c['k']} {c['caso']}: {P.SL.n(prompt)} token, codici {codici}", flush=True)
        for i, s in enumerate(P.CAMPIONI):
            if (c["k"], c["caso"], i) in fatti:
                continue
            t0 = time.time()
            r = P.post("/completion", {"prompt": prompt, "n_predict": 2500, "cache_prompt": True, **s})
            es = P.esamina(r["content"], codici)
            riga = {"k": c["k"], "caso": c["caso"], "campione": i, "parametri": s, "codici": codici,
                    "ricordi": [{"codice": v["codice"], "risposta": v["risposta"]} for v in consegne],
                    "messaggi_dopo": c.get("messaggi_dopo", 0), "trovati": es, "risposta": r["content"],
                    "secondi": round(time.time() - t0)}
            open(OUT, "a").write(json.dumps(riga, ensure_ascii=False) + "\n")
            print(f"   campione {i}: {[(x['visto'], x['esito']) for x in es]} ({riga['secondi']}s)", flush=True)


if __name__ == "__main__":
    main()
