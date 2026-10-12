#!/usr/bin/env python3
"""Taratura della soglia del richiamo (11/10): sotto soglia nessun turno del tool.
Positivi = frasi scritte a mano che parlano di un ricordo preciso del grafo (+ indizi da 2 engrammi); negativi = frasi
senza ricordo. Le frasi positive citano ricordi veri e non sono nel repo: vanno in un JSON [[frase, id_ricordo], ...]
passato come primo argomento (es. taratura_soglia.py positivi.json)."""
import json
import random
import sys

import numpy as np

import lux_engrammi as X

POSITIVI = [tuple(x) for x in json.load(open(sys.argv[1]))] if len(sys.argv) > 1 else []   # (frase, ricordo atteso)
NEGATIVI = [
    "Ieri al parco ho visto un pastore tedesco", "Che tempo fa domani a Milano?", "Mi passi la ricetta della carbonara?",
    "La Juventus ha perso di nuovo", "Devo comprare le scarpe da corsa", "Quanto costa un biglietto del treno per Firenze?",
    "Ho mal di testa da stamattina", "Il gatto della vicina ha fatto i cuccioli", "Stasera guardo un film horror",
    "Mi è finito il caffè", "Domani ho il dentista alle dieci", "Ho piantato i pomodori sul balcone",
    "Sto imparando a suonare la chitarra", "Il mio cugino si sposa a giugno", "Che differenza c'è tra lievito madre e lievito di birra?",
    "Ho perso le chiavi della macchina", "Le castagne arrosto sono la cosa migliore dell'autunno", "Quanti chilometri ci sono fino a Napoli?",
    "La lavatrice perde acqua", "Mi consigli un libro giallo?",
]

L = X.Lux()
pos = []
for q, atteso in POSITIVI:
    r = L.richiama(q, 3, 0)
    pos.append((r[0][1] if r and r[0][0] == atteso else 0.0, q, r[:1]))
random.seed(0)
for nid in random.sample([n for n in L.ricordi if L.ricordi[n]["n"] >= 6], 40):
    idx = sorted(L.per_ricordo[nid], key=lambda j: L.A[j, 1])
    e = [int(L.A[j, 3]) for j in idx]
    r = L.richiama(L.testi[e[1]] + ". " + L.testi[e[len(e) // 2 + 1]], 1, 0)
    pos.append((r[0][1] if r and r[0][0] == nid else 0.0, "indizio", r[:1]))
neg = []
for q in NEGATIVI:
    r = L.richiama(q, 1, 0)
    neg.append((r[0][1] if r else 0.0, q, r[:1]))
print("POSITIVI (punti del ricordo giusto al primo posto, 0 = non trovato):")
for p, q, r in pos[:len(POSITIVI)]:
    print(f"  {p:.3f}  {q[:70]}  -> {[(n, round(float(s), 3)) for n, s in r]}")
print("NEGATIVI (punti del primo ricordo):")
for p, q, r in neg:
    print(f"  {p:.3f}  {q[:70]}  -> {[(n, round(float(s), 3)) for n, s in r]}")
P = np.array([p for p, *_ in pos]); Ng = np.array([p for p, *_ in neg])
print(f"\nsoglia | positivi giusti consegnati ({len(P)}) | negativi consegnati per sbaglio ({len(Ng)})")
for s in [0.02, 0.05, 0.08, 0.1, 0.12, 0.15, 0.2, 0.25, 0.3]:
    print(f"  {s:.2f} | {(P >= s).sum():3d} | {(Ng >= s).sum():3d}")
