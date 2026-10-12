#!/usr/bin/env python3
"""test degli archi vivi (Lux degli engrammi): niente scritture sullo stato vero (salva_stato disattivato).
Gira sul grafo di chi lo lancia: i ricordi si passano come argomenti (gli originali erano ricordi veri, non nel repo).
  test_archi_vivi.py "<indizio del ricordo A>" <id A> <id di un ricordo mai richiamato> "<indizio del ricordo B>" "<parola del testo di B>" <id B>
L'indizio di B deve passare la soglia, l'indizio di A puo' stare sotto (si vede la soglia avvicinarsi coi richiami)."""
import sys

import lux_engrammi as X

q, nid, mai = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
q2, parola, nid2 = sys.argv[4], sys.argv[5], int(sys.argv[6])
L = X.Lux()
L.salva_stato = lambda: None
oggi = 100.0
L.U[:] = oggi
p0 = dict(L.richiama(q, 3, 0, oggi)).get(nid, 0)
assert abs(L.facilita(oggi)[nid] - 1) < 1e-9                  # appena nato: facilita' 1, taratura invariata
print(f"giorno {oggi:.0f}: punti {p0:.3f} (soglia {X.SOGLIA_RICHIAMO})")
L.rinforza(nid, oggi)
assert abs(L.facilita(oggi)[nid] - 1) < 1e-9                  # stesso giorno vissuto: crescita zero
for g in (105, 120, 160, 260, 500, 1000):   # richiami distanziati: la stabilita' cresce
    p = dict(L.richiama(q, 3, 0, g)).get(nid, 0)
    print(f"giorno {g}: punti {p:.3f}  facilita' {L.facilita(g)[nid]:.2f}  {'SOPRA' if p >= X.SOGLIA_RICHIAMO else 'sotto'} soglia")
    L.rinforza(nid, g)
assert L.facilita(1000)[nid] <= X.FACILITA_MAX + 1e-9          # tetto
print(f"ricordo {mai} mai richiamato: facilita' giorno 100 {L.facilita(100)[mai]:.2f}, giorno 1000 {L.facilita(1000)[mai]:.2f}")
assert L.facilita(1000)[mai] < L.facilita(100)[mai]            # chi non si richiama sbiadisce
c = L.consegna("Mi è finito il caffè", oggi=1000)
assert c == []                                                 # sotto soglia: niente da consegnare
L.U[L.per_ricordo[nid2]] = 1000.0                              # ricordo fresco (gli altri sono fermi al giorno 100)
S_prima = L.S[L.per_ricordo[nid2]].copy()
c = L.consegna(q2, oggi=1001)
assert c and c[0][0] == nid2 and parola in c[0][2]             # consegnato e ricomposto
assert len(c[0][1]) == 16 and L.risposta_tool(nid2).startswith(c[0][1] + "\n")   # codice sulla prima riga
assert (L.S[L.per_ricordo[nid2]] > S_prima).all()              # e rinforzato
c2 = L.consegna(q2, oggi=1001, esclusi=[c[0][1]])
assert all(x[0] != nid2 for x in c2)                           # gia' nel contesto: non si richiama di nuovo
print("test_archi_vivi: ok")
