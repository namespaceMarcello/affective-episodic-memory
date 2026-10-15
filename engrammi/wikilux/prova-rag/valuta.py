"""prova RAG (15/10): stesso indizio, tre sistemi, articolo giusto al 1o posto e fra i primi 3. Frasi esatte e riformulate."""
import json, re, sqlite3, sys, time, numpy as np
sys.path.insert(0, "/data/memoria-episodica-affettiva/engrammi")
import taglia as T, lux_engrammi as LE
D = "/data/memoria-episodica-affettiva/engrammi/wikilux/prova-rag"
z = np.load(f"{D}/wikilux.npz"); A = z["A"]
m = json.load(open(f"{D}/wikilux-meta.json")); V = np.load(f"{D}/wikilux-V.npy")
chiavi = {LE.chiave(t): i for i, t in enumerate(m["engrammi"])}
tit = {int(k): v["titolo"] for k, v in m["ricordi"].items()}; nn = {int(k): v["n"] for k, v in m["ricordi"].items()}
per_e = {}
for r, a, b in A[:, [0, 2, 3]]:
    per_e.setdefault(int(a), set()).add(int(r)); per_e.setdefault(int(b), set()).add(int(r))
blocchi = json.load(open(f"{D}/rag-blocchi.json")); R = np.load(f"{D}/rag-V.npy")
fts = sqlite3.connect(f"{D}/confronto-fts.db")

def wikilux(q):
    nodi = [t for k, t in T.taglia(q) if k == "nodo"]
    att = {}
    for t in nodi:
        if (i := chiavi.get(LE.chiave(t))) is not None: att[i] = 1.0
    if nodi:
        for row in LE.embed(nodi) @ V.T:
            for i in np.where(row >= LE.SOGLIA_SEM)[0]:
                att[int(i)] = max(att.get(int(i), 0), float((row[i] - LE.SOGLIA_SEM) / (1 - LE.SOGLIA_SEM)))
    punti = {}
    for i, a in att.items():
        ric = per_e.get(i, ())
        for r in ric: punti[r] = punti.get(r, 0) + a / np.log(1 + len(ric))
    return [tit[r] for r, _ in sorted(((r, p / np.sqrt(nn[r])) for r, p in punti.items()), key=lambda x: -x[1])[:3]]

def rag(q):
    s = R @ LE.embed([q])[0]; out = []
    for j in np.argsort(-s):
        if blocchi[j][1] not in out: out.append(blocchi[j][1])
        if len(out) == 3: break
    return out

def bm25(q):
    w = " OR ".join('"%s"' % x for x in re.findall(r"\w{3,}", q))
    return [x[0] for x in fts.execute("select titolo from r where r match ? order by bm25(r) limit 3", (w,))]

ind = json.load(open(f"{D}/indizi.json")); rif = json.load(open(f"{D}/riformulate.json"))
ris = {}
for nome, f in (("wikilux", wikilux), ("RAG", rag), ("bm25", bm25)):
    for tipo, qs in (("esatte", [x["frase"] for x in ind]), ("riformulate", rif)):
        p1 = p3 = 0; t = time.perf_counter()
        for x, q in zip(ind, qs):
            o = f(q); p1 += bool(o) and o[0] == x["titolo"]; p3 += x["titolo"] in o
        ris[f"{nome} {tipo}"] = (p1, p3, round(1000 * (time.perf_counter() - t) / len(qs)))
        print(f"{nome:8} {tipo:12} primo {p1}/{len(qs)}  primi 3 {p3}/{len(qs)}  {ris[f'{nome} {tipo}'][2]} ms", flush=True)
json.dump(ris, open(f"{D}/risultati.json", "w"), indent=1)
