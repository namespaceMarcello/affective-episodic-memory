"""aiuto della prova dei codici (16/10): legge da stdin una lista di testi, per ognuno il ricordo che il Lux
consegnerebbe (richiama, sola lettura: niente rinforzi) -> stdout [{codice, risposta} o null]"""
import json
import sys

sys.path.insert(0, "/data/memoria-episodica-affettiva/engrammi")
import lux_engrammi as X

L = X.Lux()
out = []
for t in json.load(sys.stdin):
    r = L.richiama(t, 1)
    out.append({"nid": r[0][0], "codice": L.ricordi[r[0][0]]["codice"], "risposta": L.risposta_tool([r[0][0]])} if r else None)
print(json.dumps(out, ensure_ascii=False))
