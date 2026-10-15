"""prova RAG (15/10): wikilux e RAG classico sugli stessi 500 articoli a caso, vettori bge-m3 (:8094, CPU)."""
import json, sys, time, numpy as np
sys.path.insert(0, "/data/memoria-episodica-affettiva/engrammi/wikilux"); sys.path.insert(0, "/data/memoria-episodica-affettiva/engrammi")
import wikilux as W, lux_engrammi as LE
D = "/data/memoria-episodica-affettiva/engrammi/wikilux/prova-rag"
W.D = D
W.main(0)
m = json.load(open(f"{D}/wikilux-meta.json"))
# RAG classico: blocchi di ~150 parole per articolo, paragrafi interi finche' ci stanno
blocchi = []
for nid, riga in enumerate(open(f"{D}/articoli.jsonl")):
    a, cur = json.loads(riga), []
    for p in a["paragrafi"]:
        cur.append(p)
        if sum(len(x.split()) for x in cur) >= 150:
            blocchi.append((nid, a["titolo"], " ".join(cur))); cur = []
    if cur:
        blocchi.append((nid, a["titolo"], " ".join(cur)))
json.dump(blocchi, open(f"{D}/rag-blocchi.json", "w"), ensure_ascii=False)
t = time.time(); np.save(f"{D}/rag-V.npy", LE.embed([b[2] for b in blocchi]).astype(np.float32))
print(f"RAG: {len(blocchi)} blocchi in {time.time()-t:.0f}s", flush=True)
t = time.time(); np.save(f"{D}/wikilux-V.npy", LE.embed(m["engrammi"]).astype(np.float32))
print(f"wikilux: {len(m['engrammi'])} engrammi in {time.time()-t:.0f}s", flush=True)
