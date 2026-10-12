#!/usr/bin/env python3
"""Servizio del Lux degli engrammi per il riflesso (11/10/2026). Solo 127.0.0.1:8132.
  POST /consegna  {"testo": "...", "esclusi": ["codice", ...]}
       -> {"codice": ..., "risposta": "<codice>\\n<testo>"} da mettere nel tool memorinvolontaria, oppure {} = niente
       Rinforza gli archi del ricordo consegnato (lux-<VERSIONE>/lux-stato.npz).
  GET  /salute    -> {"versione": ..., "ricordi": ..., "engrammi": ...}
Server a un filo solo: due consegne non scrivono lo stato insieme.
ponytail: libreria standard, niente framework"""
import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer

import lux_engrammi as X

PORTA = 8132
L = X.Lux()
_meta = [os.path.getmtime(X.META)]


def lux():
    """la notturna (lux_engrammi.py aggiungi) riscrive lux-meta.json per ultimo: se e' cambiato si ricarica"""
    global L
    t = os.path.getmtime(X.META)
    if t != _meta[0]:
        L, _meta[0] = X.Lux(), t
    return L


class H(BaseHTTPRequestHandler):
    def rispondi(self, oggetto, codice=200):
        corpo = json.dumps(oggetto, ensure_ascii=False).encode()
        self.send_response(codice)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(corpo)))
        self.end_headers()
        self.wfile.write(corpo)

    def do_GET(self):
        if self.path == "/salute":
            self.rispondi({"versione": X.VERSIONE, "ricordi": len(lux().ricordi), "engrammi": len(L.testi)})
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path != "/consegna":
            self.send_error(404)
            return
        try:
            d = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            c = lux().consegna(str(d.get("testo", "")), esclusi=d.get("esclusi", []))
        except Exception as e:   # il riflesso non deve mai fermarsi per il Lux: errore = nessun ricordo
            self.rispondi({"errore": repr(e)}, 500)
            return
        self.rispondi({"codice": c[0][1], "risposta": L.risposta_tool(c[0][0])} if c else {})

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    print(f"Lux {X.VERSIONE} servizio su 127.0.0.1:{PORTA} ({len(L.ricordi)} ricordi)", flush=True)
    HTTPServer(("127.0.0.1", PORTA), H).serve_forever()
