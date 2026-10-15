#!/usr/bin/env python3
"""Lux degli engrammi (11/10/2026): ricordi come indici di engrammi condivisi, legati da archi con parametri.
Il Lux originale (lux.npz, lux-meta.json) e memoria.db NON si toccano: si leggono e basta, l'uscita va in lux-<VERSIONE>/.
  costruisci   CPU: taglia i ricordi (taglia.py), engrammi unici, archi in ordine, vettori bge-m3 (:8094), firme
  aggiungi     notturna: i ricordi nuovi di memoria.db entrano senza ricostruire (i rinforzi restano)
  prova        ricomposizione di tutti i ricordi + richiamo da indizi parziali + domande di prova
  richiama "testo"   i ricordi richiamati, ricomposti (non rinforza)
  consegna "testo"   come la consegna all'agente: richiama sopra soglia, RINFORZA gli archi, ricompone
Arco = (ricordo, posizione, engramma a, engramma b, parametri fra i due): niente errori di strada sugli engrammi condivisi.
ponytail: modificatori dei parametri e propagazione non ancora usati nella scelta; via emotiva assente nel richiamo
(la firma del messaggio richiede il 35B, GPU): le firme degli engrammi sono pronte in F."""
import hashlib
import json
import os
import random
import sqlite3
import sys
import urllib.request

import numpy as np

import taglia as T

sys.path.insert(0, "/data/memoria-episodica-affettiva")
import lux as LUX   # solo le funzioni FSRS e l'orologio vissuto (lettura): il Lux originale non si tocca

# VERSIONE MAJOR.MINOR.PATCH (11/10): MAJOR = formato dei dati incompatibile (lo stato dei rinforzi non si
# porta), MINOR = ricostruzione con regole nuove (taglio, vettori) o funzione nuova, PATCH = correzione che non
# cambia i dati. Ogni versione ha la sua cartella: le vecchie restano.
VERSIONE = "1.7.0"   # 1.3.0: codice dei ricordi; 1.4.0: vettori L29 degli engrammi (lux-l29.npz); 1.5.0: aggiungi();
# 1.6.0 (15/10): vettori bge-m3 in float16 su disco e in RAM (misurato sul Lux dell'agente: scarto coseni max 1.4e-4,
# richiamo identico su 94/94 indizi); il coseno si calcola a blocchi riportati in float32
# 1.7.0 (15/10): taglio per lingua (taglia.py: le frasi inglesi sicure vanno a en_core_web_lg); costruisci()
# riusa i vettori degli engrammi gia' presenti nella versione PRECEDENTE (stesso testo = stesso vettore)
DIR = f"/data/memoria-episodica-affettiva/engrammi/lux-{VERSIONE}"
PRECEDENTE = "/data/memoria-episodica-affettiva/engrammi/lux-1.6.0"   # da qui costruisci() riprende i vettori gia' calcolati
NPZ, META, STATO = f"{DIR}/lux.npz", f"{DIR}/lux-meta.json", f"{DIR}/lux-stato.npz"
CRESCITA_PIENA = 100.0   # S/S0 a cui la facilita' tocca il tetto
FACILITA_MAX = 2.0   # tetto: un ricordo richiamato spesso arriva alla soglia con un indizio fino a 2x piu' debole
EMOZIONI = [e["en"] for e in json.load(open("/data/memoria-episodica-affettiva/emozioni.json"))]   # ordine fisso dei 51 assi
EMBEDDER = "http://127.0.0.1:8094/v1/embeddings"
SOGLIA_RICHIAMO = 0.2   # sotto: nessun ricordo, nessun turno del tool (taratura_soglia.py, 11/10: negativi max 0.177, positivi giusti min 0.268)
VTIPO = np.float16   # 1.6.0
SOGLIA_SEM = 0.8     # coseno minimo perche' un engramma del messaggio ne accenda uno del Lux
INVERSO = {}
for _k, _v in T.PREPOSIZIONI.items():   # forma canonica: la prima della tabella (di, non d'; a, non ad; tra, non fra)
    INVERSO.setdefault(_v, _k)


def codice(ts):
    """identificativo del ricordo che l'agente vede (11/10): SHA-256 del timestamp, 16 caratteri [0-9a-z]"""
    n, s = int(hashlib.sha256(repr(ts).encode()).hexdigest(), 16), ""
    for _ in range(16):
        n, r = divmod(n, 36)
        s += "0123456789abcdefghijklmnopqrstuvwxyz"[r]
    return s


def chiave(testo):
    return " ".join(testo.lower().split())


def embed(testi):
    out = []
    for i in range(0, len(testi), 64):
        r = urllib.request.Request(EMBEDDER, json.dumps({"input": testi[i:i + 64]}).encode(), {"Content-Type": "application/json"})
        out += [d["embedding"] for d in json.load(urllib.request.urlopen(r, timeout=600))["data"]]
    v = np.asarray(out, np.float32)
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)


def coseni(Q, V, blocco=65536):
    """Q (float32) contro V (float16): a blocchi di righe riportati in float32, la RAM resta quella del float16"""
    return np.concatenate([Q @ V[i:i + blocco].astype(np.float32).T for i in range(0, len(V), blocco)], axis=1)


def param_id(testo, tab):
    """parametro -> intero: preposizioni = simbolo*8+articolo (0-71), gli altri (punteggiatura, congiunzioni, che) da 100 in su"""
    c = T.codice(testo)
    if c is not None:
        return c
    k = testo.lower()
    if k not in tab:
        tab[k] = 100 + len(tab)
    return tab[k]


def param_testo(pid, altri):
    if pid < 100:
        return INVERSO[(T.SIMBOLI[pid // 8], pid % 8)]
    return altri[pid - 100]


def _inserisci(righe, eid, testi, firme_e, tab, A, par_lista, ricordi):
    """taglia i ricordi (id, testo, firma, classe, ts) e li aggiunge alle strutture: engrammi unici (chiave -> indice),
    firme per engramma, archi in ordine (ricordo, posizione, a, b, primo parametro, n parametri), parametri"""
    for nid, testo, firma, classe, ts in righe:
        pezzi = T.taglia(testo)
        f = np.asarray([json.loads(firma).get(e, 0.0) for e in EMOZIONI], np.float32) if firma else None
        nodi, legami, attesa = [], [], []
        for k, t in pezzi:
            if k == "arco":
                attesa.append(param_id(t, tab))
                continue
            kk = chiave(t)
            if kk not in eid:
                eid[kk] = len(testi)
                testi.append(t)
                firme_e.append([])
            if f is not None:
                firme_e[eid[kk]].append(f)
            nodi.append(eid[kk])
            legami.append(attesa)
            attesa = []
        ricordi[nid] = {"classe": classe, "codice": codice(ts), "n": len(nodi), "testa": legami[0] if legami else [], "coda": attesa}
        for p in range(1, len(nodi)):
            A.append((nid, p, nodi[p - 1], nodi[p], len(par_lista), len(legami[p])))
            par_lista += legami[p]
        if len(nodi) == 1:   # ricordo di un engramma solo: arco su se stesso, cosi' l'indice lo trova
            A.append((nid, 0, nodi[0], nodi[0], len(par_lista), 0))


def costruisci():
    os.makedirs(DIR, exist_ok=True)
    if os.path.exists(STATO):   # gli archi cambiano numerazione: lo stato dei rinforzi non varrebbe piu'
        sys.exit(f"{STATO} esiste: ricostruire cancella i rinforzi degli archi. Spostalo a mano se e' voluto.")
    c = sqlite3.connect(f"file:{T.DB}?mode=ro", uri=True)
    righe = c.execute("select id, testo, firma, classe, ts from nodi where testo != '' order by id").fetchall()
    eid, testi, firme_e, tab, A, par_lista, ricordi = {}, [], [], {}, [], [], {}
    _inserisci(righe, eid, testi, firme_e, tab, A, par_lista, ricordi)
    print(f"{len(righe)} ricordi -> {len(testi)} engrammi unici, {len(A)} archi, {len(tab)} parametri non preposizione", flush=True)
    V = np.zeros((len(testi), 1024), VTIPO)
    vecchi = {}
    if os.path.exists(f"{PRECEDENTE}/lux.npz"):
        zp, mp = np.load(f"{PRECEDENTE}/lux.npz"), json.load(open(f"{PRECEDENTE}/lux-meta.json"))
        vecchi = {t: j for j, t in enumerate(mp["engrammi"])}
        for i, t in enumerate(testi):
            if t in vecchi:
                V[i] = zp["V"][vecchi[t]]
    nuovi = [i for i, t in enumerate(testi) if t not in vecchi]
    print(f"vettori: {len(testi) - len(nuovi)} ripresi da {PRECEDENTE}, {len(nuovi)} da calcolare", flush=True)
    if nuovi:
        V[nuovi] = embed([testi[i] for i in nuovi]).astype(VTIPO)
    F = np.stack([np.mean(x, 0) if x else np.zeros(51, np.float32) for x in firme_e])
    A = np.asarray(A, np.int64)
    np.savez(NPZ, V=V, F=F, A=A, P=np.asarray(par_lista, np.int32))
    altri = [k for k, _ in sorted(tab.items(), key=lambda x: x[1])]
    json.dump({"engrammi": testi, "altri_parametri": altri, "ricordi": ricordi}, open(META, "w"), ensure_ascii=False)
    print(f"scritto {NPZ} e {META}")


def aggiungi():
    """1.5.0 (notturna, 12/10): i ricordi nuovi di memoria.db entrano nel Lux senza ricostruire. Engrammi nuovi in coda
    (vettori bge-m3, firme), quelli gia' noti si condividono, archi nuovi in coda con la stabilita' iniziale; i rinforzi
    degli archi esistenti restano. Scrittura atomica: il servizio ricarica quando cambia lux-meta.json.
    ponytail: la firma di un engramma gia' noto non si ricalcola con i ricordi nuovi (servirebbero i conteggi);
    lux-l29.npz copre solo gli engrammi della costruzione (non usato nel richiamo)."""
    z, m = np.load(NPZ), json.load(open(META))
    ricordi = {int(k): v for k, v in m["ricordi"].items()}
    c = sqlite3.connect(f"file:{T.DB}?mode=ro", uri=True)
    righe = [r for r in c.execute("select id, testo, firma, classe, ts from nodi where testo != '' order by id") if r[0] not in ricordi]
    if not righe:
        print("nessun ricordo nuovo")
        return
    testi = list(m["engrammi"])
    eid = {chiave(t): i for i, t in enumerate(testi)}
    tab = {k: 100 + i for i, k in enumerate(m["altri_parametri"])}
    n_e, n_a = len(testi), len(z["A"])
    firme_e = [[] for _ in testi]
    A, par_lista = [], list(z["P"])
    _inserisci(righe, eid, testi, firme_e, tab, A, par_lista, ricordi)
    nuovi = testi[n_e:]
    V = np.concatenate([z["V"].astype(VTIPO), embed(nuovi).astype(VTIPO)]) if nuovi else z["V"].astype(VTIPO)
    F = np.concatenate([z["F"], np.stack([np.mean(x, 0) if x else np.zeros(51, np.float32) for x in firme_e[n_e:]])]) if nuovi else z["F"]
    A = np.concatenate([z["A"], np.asarray(A, np.int64).reshape(-1, 6)])
    if os.path.exists(STATO):   # archi nuovi: stabilita' iniziale dalla firma del ricordo, nati oggi (vissuto)
        s = np.load(STATO)
        firme = {r[0]: json.loads(r[2]) for r in righe if r[2]}
        s0 = np.asarray([LUX.Lux.stabilita_iniziale(np.asarray([firme[int(r)].get(e, 0.0) for e in EMOZIONI]))
                         if int(r) in firme else LUX.S_BASE for r in A[n_a:, 0]], np.float64)
        oggi = np.full(len(s0), float(LUX.orologio()))
        np.savez(STATO + ".tmp.npz", S=np.concatenate([s["S"], s0]), S0=np.concatenate([s["S0"], s0]), U=np.concatenate([s["U"], oggi]))
    np.savez(NPZ + ".tmp.npz", V=V, F=F, A=A, P=np.asarray(par_lista, np.int32))
    altri = [k for k, _ in sorted(tab.items(), key=lambda x: x[1])]
    json.dump({"engrammi": testi, "altri_parametri": altri, "ricordi": ricordi}, open(META + ".tmp", "w"), ensure_ascii=False)
    os.replace(NPZ + ".tmp.npz", NPZ)
    if os.path.exists(STATO + ".tmp.npz"):
        os.replace(STATO + ".tmp.npz", STATO)
    os.replace(META + ".tmp", META)   # per ultimo: e' il segnale di ricarica del servizio
    print(f"{len(righe)} ricordi nuovi, {len(nuovi)} engrammi nuovi, {len(A) - n_a} archi nuovi")


class Lux:
    def __init__(self):
        z = np.load(NPZ)
        self.V, self.F, self.A, self.P = z["V"].astype(VTIPO, copy=False), z["F"], z["A"], z["P"]
        m = json.load(open(META))
        self.testi, self.altri, self.ricordi = m["engrammi"], m["altri_parametri"], {int(k): v for k, v in m["ricordi"].items()}
        self.chiavi = {chiave(t): i for i, t in enumerate(self.testi)}
        self.da_codice = {r["codice"]: n for n, r in self.ricordi.items()}
        # 1.4.0: vettori L29 degli engrammi (vettori_l29.py), stesso spazio di addr-sem; per la via L29 serve lo
        # stato L29 del messaggio, che legge luxifer (35B, GPU): ancora non usati nel richiamo
        f = f"{DIR}/lux-l29.npz"
        self.L29, self.ha_l29 = (np.load(f)["L"], np.load(f)["presente"]) if os.path.exists(f) else (None, None)
        self.carica_stato()
        self.per_ricordo = {}
        for j, r in enumerate(self.A[:, 0]):
            self.per_ricordo.setdefault(int(r), []).append(j)
        self.per_engramma = {}
        for j, (a, b) in enumerate(self.A[:, 2:4]):
            self.per_engramma.setdefault(int(a), set()).add(j)
            self.per_engramma.setdefault(int(b), set()).add(j)

    # ---------- archi vivi (11/10): stabilita' FSRS per arco, come nel Lux originale ----------
    def carica_stato(self):
        """S (stabilita') e U (ultimo giorno vissuto) per arco. Al primo avvio: S0 dalla firma del ricordo
        (LUX.stabilita_iniziale: fino a 3x se emotivamente forte), U = oggi vissuto"""
        if os.path.exists(STATO):
            z = np.load(STATO)
            self.S, self.S0, self.U = z["S"], z["S0"], z["U"]
            return
        c = sqlite3.connect(f"file:{T.DB}?mode=ro", uri=True)
        firme = {nid: json.loads(f) for nid, f in c.execute("select id, firma from nodi where firma is not null")}
        s0 = {nid: LUX.Lux.stabilita_iniziale(np.asarray([f.get(e, 0.0) for e in EMOZIONI])) for nid, f in firme.items()}
        self.S = np.asarray([s0.get(int(r), LUX.S_BASE) for r in self.A[:, 0]], np.float64)
        self.S0 = self.S.copy()
        self.U = np.full(len(self.A), float(LUX.orologio()))

    def salva_stato(self):
        np.savez(STATO + ".tmp.npz", S=self.S, S0=self.S0, U=self.U)
        os.replace(STATO + ".tmp.npz", STATO)

    def facilita_archi(self, oggi=None):
        """per arco: ritenzione R (cala se non si richiama) x crescita della stabilita' dalla nascita, in scala
        logaritmica fino al tetto FACILITA_MAX quando S = CRESCITA_PIENA x S0 (con FSRS un richiamo distanziato
        moltiplica S per ~5: il tetto arriva in 3-4 richiami). Appena nato = 1: la soglia tarata l'11/10 resta valida"""
        oggi = LUX.orologio() if oggi is None else oggi
        R = LUX.curva(np.maximum(oggi - self.U, 0.0), self.S)
        crescita = np.minimum(1.0, np.log(self.S / self.S0) / np.log(CRESCITA_PIENA))
        return R * (1 + (FACILITA_MAX - 1) * crescita)

    def facilita(self, oggi=None):
        """per ricordo: media della facilita' dei suoi archi -> moltiplica il punteggio del richiamo"""
        f = self.facilita_archi(oggi)
        idx = np.unique(self.A[:, 0], return_inverse=True)
        media = np.bincount(idx[1], f) / np.bincount(idx[1])
        return dict(zip(idx[0].tolist(), media))

    def rinforza(self, nid, oggi=None):
        """richiamo riuscito del ricordo nid (consegnato all'agente): ogni suo arco cresce con FSRS (tanto piu' quanto stava
        svanendo; stesso giorno vissuto = crescita zero) e riparte da oggi. Salva lo stato."""
        oggi = LUX.orologio() if oggi is None else oggi
        js = self.per_ricordo.get(nid, [])
        R = LUX.curva(np.maximum(oggi - self.U[js], 0.0), self.S[js])
        self.S[js] = [LUX.cresci(float(s), float(r)) for s, r in zip(self.S[js], R)]
        self.U[js] = oggi
        self.salva_stato()

    def consegna(self, testo, quanti=1, oggi=None, esclusi=()):
        """L'UNICA porta per dare ricordi all'agente (richiamo automatico, 11/10): richiama, rinforza gli archi dei
        ricordi consegnati, restituisce [(id, testo ricomposto)]. Lista vuota = nessun turno del tool.
        esclusi = codici dei ricordi gia' consegnati nella conversazione: si passa al successivo sopra soglia.
        La pagina 3D e le prove usano richiama(), che non rinforza."""
        via = {self.da_codice[c] for c in esclusi if c in self.da_codice}   # gia' nel contesto della conversazione
        scelti = [x for x in self.richiama(testo, quanti + len(via), oggi=oggi) if x[0] not in via][:quanti]
        for nid, _ in scelti:
            self.rinforza(nid, oggi)
        return [(nid, self.ricordi[nid]["codice"], self.ricomponi(nid)) for nid, _ in scelti]

    def risposta_tool(self, nid):
        """il testo che l'agente riceve nel tool memorinvolontaria (11/10): codice sulla prima riga, poi il testo nudo"""
        return self.ricordi[nid]["codice"] + "\n" + self.ricomponi(nid)

    def ricomponi(self, nid):
        arcs = sorted(self.per_ricordo.get(nid, []), key=lambda j: self.A[j, 1])
        info = self.ricordi[nid]
        pezzi = [("arco", param_testo(p, self.altri)) for p in info["testa"]]
        for k, j in enumerate(arcs):
            _, pos, a, b, p0, n = self.A[j]
            if k == 0:
                pezzi.append(("nodo", self.testi[a]))
            if pos == 0:
                continue
            pezzi += [("arco", param_testo(int(p), self.altri)) for p in self.P[p0:p0 + n]]
            pezzi.append(("nodo", self.testi[b]))
        pezzi += [("arco", param_testo(p, self.altri)) for p in info["coda"]]
        out = ""
        for k, t in pezzi:
            attacca = k == "arco" and (t in ",.;:!?)…" or t.startswith("...")) or out.endswith(("(", "'", "’"))
            out += ("" if attacca or not out else " ") + t
        return out

    def accendi(self, testo):
        """innesco: via testuale (stesso engramma = 1) e via semantica (bge-m3, coseno sopra soglia riscalato 0-1)"""
        att = {}
        nodi = [t for k, t in T.taglia(testo) if k == "nodo"]
        if not nodi:
            return att
        for t in nodi:
            i = self.chiavi.get(chiave(t))
            if i is not None:
                att[i] = 1.0
        S = coseni(embed(nodi), self.V)
        for row in S:
            for i in np.where(row >= SOGLIA_SEM)[0]:
                att[int(i)] = max(att.get(int(i), 0), float((row[i] - SOGLIA_SEM) / (1 - SOGLIA_SEM)))
        return att

    def richiama(self, testo, quanti=3, soglia=SOGLIA_RICHIAMO, oggi=None):
        """scelta: per ogni indice di ricordo la somma delle attivazioni dei suoi engrammi, pesate 1/log(1+ricordi che
        usano l'engramma) (gli engrammi comuni contano poco) e divise per sqrt(engrammi del ricordo) (i lunghi non vincono
        sempre), poi moltiplicate per la facilita' del ricordo (archi vivi). Tarato l'11/10 su 40 indizi: soglia 0.8, nessuna propagazione nella scelta (abbassava da 35 a 30)."""
        punti = {}
        for i, a in self.accendi(testo).items():
            ric = {int(self.A[j, 0]) for j in self.per_engramma.get(i, ())}
            for r in ric:
                punti[r] = punti.get(r, 0) + a / np.log(1 + len(ric))
        fac = self.facilita(oggi)
        for r in punti:
            punti[r] *= fac[r] / np.sqrt(self.ricordi[r]["n"])
        return [x for x in sorted(punti.items(), key=lambda x: -x[1])[:quanti] if x[1] >= soglia]


def prova():
    L = Lux()
    c = sqlite3.connect(f"file:{T.DB}?mode=ro", uri=True)
    orig = dict(c.execute("select id, testo from nodi where testo != ''"))
    norm = lambda s: "".join(T.FORMATO.sub("", s).split())
    ok = [nid for nid in L.ricordi if norm(L.ricomponi(nid)) == norm(orig[nid])]
    ok_min = [nid for nid in L.ricordi if norm(L.ricomponi(nid)).lower() == norm(orig[nid]).lower()]
    print(f"ricomposizione identica (spazi a parte): {len(ok)}/{len(L.ricordi)}, maiuscole a parte: {len(ok_min)}/{len(L.ricordi)}")
    random.seed(0)
    prove = random.sample([n for n in L.ricordi if L.ricordi[n]["n"] >= 6], 40)
    colpiti = 0
    for nid in prove:   # indizio parziale: due engrammi del ricordo, non vicini
        idx = sorted(L.per_ricordo[nid], key=lambda j: L.A[j, 1])
        e = [int(L.A[j, 3]) for j in idx]
        indizio = L.testi[e[1]] + ". " + L.testi[e[len(e) // 2 + 1]]
        r = L.richiama(indizio, 1, 0)
        colpiti += bool(r) and r[0][0] == nid
    print(f"richiamo da due engrammi del ricordo: {colpiti}/{len(prove)} al primo posto")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "costruisci":
        costruisci()
    elif cmd == "aggiungi":
        aggiungi()
    elif cmd == "prova":
        prova()
    elif cmd == "consegna":   # come la consegna vera: rinforza e scrive lux-stato.npz
        L = Lux()
        for nid, cod, testo in L.consegna(sys.argv[2]):
            print(f"== ricordo {nid}\n{L.risposta_tool(nid)}")
    elif cmd == "richiama":
        L = Lux()
        for nid, p in L.richiama(sys.argv[2]):
            print(f"\n== ricordo {nid}  ({p:.2f})\n{L.ricomponi(nid)}")
