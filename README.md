# 🎧 Bio Responsive Soundscape
**Bio Responsive Soundscape è un server backend avanzato che genera musica adattiva in tempo reale basata sui dati biometrici dell'utente. Sfruttando un LLM per la traduzione semantica degli stati fisiologici e il modello MusicGen di Meta per la sintesi audio, il sistema crea un flusso continuo di musica che si adatta alle performance atletiche durante la corsa, il sollevamento pesi o lo yoga.**

## 🚀 Caratteristiche Principali
* Generazione Audio Continua: Utilizza la tecnica di continuation di MusicGen per generare chunk audio sovrapposti, garantendo transizioni fluide e senza interruzioni tra un segmento e l'altro.

* Traduzione Biometrica (LLM): Converte dati in tempo reale (battito cardiaco, cadenza, trend di performance) in prompt musicali tecnici, adattando l'energia e il ritmo alla fase dell'allenamento.

* Gestione Sessioni Asincrona: Sviluppato con FastAPI, il server delega l'inferenza pesante (LLM e generazione audio) a thread in background, mantenendo gli endpoint reattivi tramite un sistema di polling dello stato.

* Sistema di Caching Intelligente (AudioCache): Include un sistema di cache condivisa thread-safe per evitare la rigenerazione di prompt identici, ottimizzando drasticamente i tempi di risposta (limite configurabile, default: 500MB).

* Multi-Paradigma: Supporta logiche di generazione distinte:

    * 🏃 Running: Flusso singolo con energia modulabile.

  * 🏋️ Weightlifting: Flusso doppio (traccia aggressiva per il lifting e traccia d'atmosfera senza batteria per il resting).

  * 🧘 Yoga: Flusso ambientale focalizzato sull'evoluzione delle frequenze, rigorosamente senza percussioni.

## 🛠️ Stack Tecnologico
* Core API: FastAPI, Uvicorn, Pydantic

* AI / Inferenza: PyTorch (con supporto CUDA), HuggingFace Transformers (facebook/musicgen-small)

* LLM Integration: Google Generative AI (models/gemma-4-26b-a4b-it)

* Elaborazione Audio: SciPy, NumPy

* Networking: Pyngrok (per l'esposizione del server locale)

## 📁 Struttura del Progetto
* server.py: Entrypoint dell'applicazione FastAPI. Gestisce il routing, le code di sessione in background, il download dei file e l'indicizzazione della cache audio.

* llm_gen.py: Modulo dedicato all'interfacciamento con le API di Google. Contiene i system prompt specifici per ogni attività e parsa le risposte JSON per estrarre la categoria di cache e i musicgen_prompt.

* music_gen.py: Core engine per la sintesi audio. Gestisce il caricamento del modello su GPU/CPU, la generazione zero-shot e la continuazione basata sul contesto audio precedente (ParadigmaRunning, ParadigmaWeightlifting, ParadigmaYoga).

## ⚙️ Installazione e Avvio
1. Clona il repository e installa le dipendenze:
Assicurati di avere un ambiente Python con supporto PyTorch (preferibilmente configurato per CUDA se disponi di una GPU dedicata).

```bash
pip install fastapi uvicorn pydantic scipy numpy torch transformers google-generativeai pyngrok
```
2. Configura le API Key:
Nel file api_token.py (da creare se non presente), inserisci la tua chiave per Gemini:
```bash
Python
GeminiAPI = "CHIAVE PER GEMINI"
NgrokAPI = "CHIAVE PER NGROK"
HuggingFaceAPI = "CHIAVE HUGGINGFACE"
modello1 = "models/gemma-4-26b-a4b-it" //LLM (DI GOOGLE) CHE SI VUOLE UTILIZZARE
modello2 = "facebook/musicgen-small" //MODELLO MUSICGEN (DI META) CHE SI VUOLE UTILIZZARE
```
Nota: Inserisci il tuo auth token personale di ngrok all'interno di server.py se intendi esporre il server pubblicamente.

3. Avvia il server:
Esegui direttamente il file principale. Il server avvierà automaticamente Uvicorn e aprirà un tunnel ngrok.

```bash
python server.py
```
## 📡 API Reference Inizializza una Sessione
1. Crea una nuova sessione e avvia la generazione del primo chunk audio in background.
    ```bash
    Endpoint: POST /session/start
    ```
    Payload (Esempio Running):
    ```bash
     
    {
      "payload": {
        "activity": "running",
        "user_initial_intent": "Voglio spingere al massimo con della techno.",
        "current_avg_hr": 155,
        "cadence_spm": 160,
        "performance_trend": "accelerazione",
        "target_bpm": 160
      }
    }
    ```
2. Polling dello Stato
Verifica se l'audio per una determinata sessione è pronto per il download.
    ```bash
    Endpoint: GET /session/{session_id}/status
    ```
   
    Risposte previste: generating | ready | error


3. Aggiorna Sessione
Invia nuovi dati biometrici per influenzare la generazione del chunk successivo.
    ```bash
    Endpoint: POST /session/{session_id}/update
    ```
    Payload: Stessa struttura di /start. Aggiorna dinamicamente l'energia e il genere musicale.


4. Scarica Audio
Ottieni il file generato. Se l'attività prevede tracce multiple (come il sollevamento pesi), restituisce un archivio .zip contenente i file separati.
    ```bash
    Endpoint: GET /session/{session_id}/audio
    ```
   
5. Pulizia Sessione
Elimina la sessione dalla memoria e pulisce tutti i file .wav temporanei associati dal disco (preservando però l'utilissima AudioCache).
    ```bash
    Endpoint: DELETE /session/{session_id}
    ```
   
## ⚠️ Limitazioni e Note
* Memoria GPU: MusicGen richiede una quantità moderata di VRAM. Il caricamento in float16 è abilitato di default su CUDA per ottimizzare le risorse.

* Tempi di Inferenza: A seconda dell'hardware, la generazione di 22 secondi di audio può richiedere da pochi secondi a diversi minuti. Il sistema di polling asincrono maschera questa latenza al client.

* Durata Massima Modello: Il modello base ha un hard-limit di 30 secondi per singolo prompt (1500 tokens). L'architettura aggira il problema gestendo l'overlap temporale tra i chunk per brani potenzialmente infiniti.