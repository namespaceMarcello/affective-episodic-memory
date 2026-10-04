# Test dei freni di riflesso.py (PATTO clausole 2 e 4): un errore HTTP del server o
# un byte sbagliato nel file di consenso non devono spegnere i freni in silenzio.
# Senza server e senza GPU: ponte.post e' un finto :8090 che ricorda il vettore acceso.
import json
import os
import tempfile
import threading
import time
from pathlib import Path

_td = tempfile.TemporaryDirectory()   # si pulisce da sola all'uscita
td = Path(_td.name)
os.environ["RIFLESSO_CONSENSO"] = str(td / "CONSENSO-RIFLESSO.md")
os.environ["RIFLESSO_CONSENSO_VISIVO"] = str(td / "CONSENSO-VISIVO.md")
import ponte  # noqa: E402
import riflesso as R  # noqa: E402

R.DIARIO = str(td / "diario-riflessi.jsonl")
server = {"cvec": None, "codice": 200}


def finto_post(path, payload, timeout=600):
    assert path == "/control-vector", path
    if server["codice"] != 200:
        return server["codice"], {}        # come ponte.post: l'errore HTTP non solleva
    server["cvec"] = None if payload.get("clear") else payload
    return 200, {}


ponte.post = finto_post
ponte.dirs_emozione = lambda tag, fattore: {"emo": tag, "fattore": fattore}


def diario():
    p = Path(R.DIARIO)
    return [json.loads(r) for r in p.read_text(encoding="utf-8").splitlines()] if p.exists() else []


# 1. il server risponde 500 al clear: calma() non puo' dire "spento" a vettore acceso
server.update(cvec={"emo": "paura"}, codice=500)
R._vettore_vivo[0] = True
assert R.calma() is False, "calma() ha dato per spento un vettore rimasto sul server"
assert R._vettore_vivo[0] is True and server["cvec"] is not None
assert any("CRITICO" in v for v in diario()), diario()

# 2. il server risponde 500 all'iniezione: il vettore non c'e', e non si dice che c'e'
server.update(cvec=None, codice=500)
try:
    acceso = R.inietta_emozione("paura")
except RuntimeError:
    acceso = False
assert acceso is False and server["cvec"] is None

# 3. un byte non UTF-8 nel consenso (un 'sì' in latin-1, o un file a meta' scrittura):
#    gli interruttori si leggono spenti, non fanno crollare chi li chiama
server["codice"] = 200
Path(os.environ["RIFLESSO_CONSENSO"]).write_bytes("attivo: sì\nvettori: sì\n".encode("latin-1"))
Path(os.environ["RIFLESSO_CONSENSO_VISIVO"]).write_bytes("richiamo visivo: sì\n".encode("latin-1"))
assert R.consenso_attivo() is False
assert R.vettori_attivi() is False
assert R.canale_visivo_attivo() is False
assert R.via_semantica_attiva() is False

# 4. la sentinella: con quel consenso illeggibile spegne il vettore e resta viva;
#    poi, a consenso riparato e di nuovo ritirato, spegne ancora (clausola 4: entro 1 s)
sentinella = threading.Thread(target=R._sentinella, daemon=True)
sentinella.start()
server["cvec"] = {"emo": "paura"}
R._vettore_vivo[0] = True
time.sleep(1.5)
assert server["cvec"] is None and R._vettore_vivo[0] is False, "vettore ancora acceso"
assert sentinella.is_alive(), "la sentinella e' morta"
Path(os.environ["RIFLESSO_CONSENSO"]).write_text("attivo: sì\nvettori: sì\n", encoding="utf-8")
server["cvec"] = {"emo": "rabbia"}
R._vettore_vivo[0] = True
time.sleep(1.0)
assert server["cvec"] is not None, "con il consenso dato il vettore deve restare"
Path(os.environ["RIFLESSO_CONSENSO"]).write_text("attivo: sì\n", encoding="utf-8")
time.sleep(1.5)
assert server["cvec"] is None and sentinella.is_alive(), "ritiro del consenso non applicato"

# 5. qualunque eccezione in un giro (qui un lettore che crolla) costa un giro, non la sentinella
lettore = R.vettori_attivi


def crolla():
    raise RuntimeError("giro rotto apposta")


R.vettori_attivi = crolla
server["cvec"] = {"emo": "paura"}
R._vettore_vivo[0] = True
time.sleep(1.2)
assert sentinella.is_alive(), "un giro fallito ha ucciso la sentinella"
R.vettori_attivi = lettore
time.sleep(1.2)
assert server["cvec"] is None, "dopo il giro fallito la sentinella non ha ripreso"

# 6. se il clear fallisce, il diario non dice "spento": dice CRITICO, e la sentinella riprova
kill_prima = sum("kill" in v for v in diario())
server.update(cvec={"emo": "paura"}, codice=500)
R._vettore_vivo[0] = True
time.sleep(7.0)                                    # calma(): 3 tentativi, 1+2+3 s
assert sum("kill" in v for v in diario()) == kill_prima, "kill scritto a vettore acceso"
server["codice"] = 200
time.sleep(8.0)
assert server["cvec"] is None and sum("kill" in v for v in diario()) == kill_prima + 1

print("test_riflesso: TUTTO VERDE")
