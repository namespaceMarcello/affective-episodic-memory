"""prova a tre sui ricordi dell'agente (15/10): Lux 1.5.0 (richiama vero, soglia 0) contro RAG classico e bm25.
Indizi: le 14 frasi libere di taratura_soglia.py (scritte da CC, parole diverse dal ricordo), 40 indizi da due
engrammi (come la taratura), 40 frasi esatte copiate dai ricordi. Sola lettura: memoria.db e il Lux non si toccano."""
import json, os, random, re, sqlite3, sys, time, numpy as np
sys.path.insert(0, "/data/memoria-episodica-affettiva/engrammi")
import lux_engrammi as X, taratura_soglia as TS  # noqa (TS stampa la taratura all'import)
D = os.path.dirname(os.path.abspath(__file__))
L = TS.L
c = sqlite3.connect(f"file:{X.T.DB}?mode=ro", uri=True)
ric = {i: t for i, t in c.execute("select id, testo from nodi where testo != ''") if i in L.ricordi}
# RAG: blocchi di ~150 parole per ricordo, frasi intere
blocchi = []
for i, t in ric.items():
    cur = []
    for s in re.split(r"(?<=[.!?])\s+", t):
        cur.append(s)
        if sum(len(x.split()) for x in cur) >= 150:
            blocchi.append((i, " ".join(cur))); cur = []
    if cur: blocchi.append((i, " ".join(cur)))
f = f"{D}/rag-V.npy"
if not os.path.exists(f):
    t0 = time.time(); np.save(f, X.embed([b[1] for b in blocchi])); print(f"RAG: {len(blocchi)} blocchi in {time.time()-t0:.0f}s", flush=True)
R = np.load(f)
db = sqlite3.connect(":memory:"); db.execute("create virtual table r using fts5(id unindexed, testo)")
db.executemany("insert into r values(?, ?)", ric.items())

def lux(q): return [n for n, _ in L.richiama(q, 3, 0)]
def rag(q):
    s, out = R @ X.embed([q])[0], []
    for j in np.argsort(-s):
        if blocchi[j][0] not in out: out.append(blocchi[j][0])
        if len(out) == 3: return out
def bm25(q):
    w = " OR ".join('"%s"' % x for x in re.findall(r"\w{3,}", q))
    return [int(x[0]) for x in db.execute("select id from r where r match ? order by bm25(r) limit 3", (w,))] if w else []

random.seed(0)
due = []
for nid in random.sample([n for n in L.ricordi if L.ricordi[n]["n"] >= 6], 40):
    idx = sorted(L.per_ricordo[nid], key=lambda j: L.A[j, 1]); e = [int(L.A[j, 3]) for j in idx]
    due.append((L.testi[e[1]] + ". " + L.testi[e[len(e) // 2 + 1]], nid))
random.seed(1)
esatte = []
for nid in random.sample(list(ric), 40):
    fr = [s for s in re.split(r"(?<=[.!?])\s+", ric[nid]) if 40 < len(s) < 220] or [ric[nid][:200]]
    esatte.append((random.choice(fr), nid))
gruppi = {"frasi libere": TS.POSITIVI, "due engrammi": due, "frasi esatte": esatte}
ris = {}
print()
for nome, fn in (("Lux", lux), ("RAG", rag), ("bm25", bm25)):
    for g, qs in gruppi.items():
        p1 = p3 = 0; t = time.perf_counter()
        for q, atteso in qs:
            o = fn(q) or []; p1 += bool(o) and o[0] == atteso; p3 += atteso in o
        ms = round(1000 * (time.perf_counter() - t) / len(qs))
        ris[f"{nome} {g}"] = (p1, p3, len(qs), ms)
        print(f"{nome:5} {g:13} primo {p1:2}/{len(qs)}  primi 3 {p3:2}/{len(qs)}  {ms} ms", flush=True)
json.dump(ris, open(f"{D}/risultati.json", "w"), indent=1)
