# Lux degli engrammi: piano delle fasi 2 e 3 (dopo la 1.11.0)

La 1.11.0 contiene la fase 1, le correzioni: un ricordo che ne corregge un altro, legato da un arco con i parametri
`{corregge}{codice}`, consegnato in catena nella stessa risposta del tool, con l'originale barrato e mai piu' rinforzato.
Questo documento descrive le due fasi successive. Regola comune: ogni dato nuovo sta dentro le strutture del Lux
(`lux.npz`, engrammi, archi, stato degli archi), mai in file a parte.

## Fase 2. L'emozione che cambia col richiamo (riconsolidamento)

**Dipende dal test delle emozioni nei layer.** La firma emotiva di un ricordo diventa la dose del vettore di steering che
colora il ricordo alla consegna. Il modello accetta le dosi solo dentro un certo intervallo, e alcuni layer vanno evitati
secondo la lunghezza del contesto e l'emozione. Tutti i numeri di questa fase (impatto, dissipazione, tetti) si esprimono
nelle unita' della dose e si ricavano da quel test, con la funzione dose per emozione che lo chiude.

1. **Firma corrente.** La firma di ogni ricordo vive in `lux.npz` (chiave `emo_F`); all'inizio e' quella di memoria.db.
   Le firme degli engrammi (`F`) si ricalcolano da li'.
2. **Impatto alla nascita.** Ogni emozione ha un peso (tabella in `salienza/emozioni-scienza.md`): le negative ad alta
   attivazione nascono piu' forti. La salienza usa la firma pesata.
3. **Rinforzo emotivo, come nella mente umana.** Ogni richiamo:
   - **potenzia** l'emozione specifica del ricordo: il valore si avvicina al tetto con passi sempre piu' piccoli e non lo
     tocca mai; il tetto sta almeno 0,02 sotto la soglia di eco, cioe' la dose a cui l'emozione esce come parola
     nell'output del modello;
   - **la stabilizza**: piu' rinforzi ha preso, piu' lentamente quell'emozione sbiadisce.
4. **La firma del momento.** Il tagging calcola gia' la firma di ogni messaggio della giornata. Per ogni consegna del
   diario dei riflessi scrive nel diario la firma della prima risposta dell'agente dopo il ricordo (passaggio di una
   notte; il dato del ricordo resta nel Lux).
5. **Passo notturno `riconsolida`** (dopo `aggiungi`): la firma si sposta verso quella del momento,
   firma <- firma + alfa x (momento - firma), alfa piu' piccolo per i ricordi stabili (alfa = 0,2 / (1 + log(S/S0)))
   e un tetto per notte. Anche qui i passi si riducono avvicinandosi ai limiti.
6. **Dissipazione (Fading Affect Bias).** Ogni notte la parte di firma di ogni emozione cala secondo la tabella: le
   negative prima, le positive lentamente, le neutre ferme; colpa, vergogna e rimorso 0,25% a notte. Lo sbiadimento
   rallenta avvicinandosi allo zero: nessuno dei due estremi si raggiunge.
7. **Effetto sulla forza.** Dalla firma nuova si ricalcola S0; S si scala nella stessa proporzione, cosi' i rinforzi
   gia' presi restano.
8. **Rodaggio di 30 giorni** per tutti i cambiamenti: giorni 0-10 al 10% dell'effetto pieno, 10-20 al 30%, 20-30 al
   70%, poi 100%. La data di partenza sta in `lux.npz`. Copia dello stato prima di ogni notte.
9. **Errori.** Se `riconsolida` fallisce la notturna prosegue. Il vettore del riflesso continua a usare l'`emo_tag` di
   memoria.db finche' non si decide altrimenti.
10. **Permanenza** (durata dell'episodio emotivo): fuori da questa fase. Riguarda lo stato d'animo, non la memoria;
    se servira' andra' nell'omeostato.

**Da decidere:** la divisione delle 51 emozioni in negative, positive e neutre (proposta in `salienza/emozioni-scienza.md`);
i numeri di impatto e dissipazione nelle unita' della dose, dopo il test per layer.

## Fase 3. Fonti

1. **Le fonti sono testi:** quello che l'agente legge online (ricerca, pagine) e quello che gli scrive l'utente.
2. **La fonte si riconosce da sola dalla provenienza nel contesto**, momento per momento: un messaggio dell'utente, il
   risultato di uno strumento con il suo indirizzo. L'agente non la sceglie e non la dichiara: il riflesso la ricava dal
   punto del contesto da cui arriva la notizia.
3. **La fonte e' un engramma** legato ai ricordi che ha prodotto. La fiducia e' la forza dei suoi archi: cresce quando
   una sua notizia viene confermata, cala quando viene corretta.
4. **Chi corregge.** L'utente e l'agente. Una fonte web diventa "certa" quando la sua fiducia supera una soglia: solo
   allora una sua notizia puo' produrre una correzione. L'utente e' certa dall'inizio.
5. **Strumento di correzione per l'agente** (codice del ricordo e testo; la fonte la aggiunge il riflesso): chiama
   `/correggi` del servizio del Lux, gia' presente dalla 1.11.0. Il testo della descrizione dello strumento lo decide
   l'utente prima che entri nel codice.

**Da decidere:** il testo dello strumento di correzione; la soglia di fiducia; se le correzioni dell'agente entrano
subito o nella notte.

## Verifica (per entrambe le fasi)

- Con il rodaggio a zero e senza fonti, risultati identici alla versione precedente.
- In una copia del Lux: notti finte con firme finte; rinforzo che si avvicina al tetto senza toccarlo; dissipazione per
  emozione; rodaggio; fonti che guadagnano e perdono fiducia; nessun cambiamento sugli altri ricordi.
- `verifica_post_versioning.py` sulla cartella nuova.
