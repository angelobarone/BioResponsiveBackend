import torch
import scipy.io.wavfile
import numpy as np
import os
import time
import threading
from transformers import AutoProcessor, MusicgenForConditionalGeneration

import api_token

# INIZIALIZZAZIONE GLOBALE
# Creiamo una cartella per salvare i brani generati durante l'allenamento
OUT_DIR = "output_allenamento"
os.makedirs(OUT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Accelerazione hardware attivata sul dispositivo: {device}")

model_id = api_token.modello2

print("Caricamento del modello... (potrebbe richiedere un po' di tempo al primo avvio)")
if device == "cuda":
    model = MusicgenForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,  # Carica in metà precisione
        device_map="auto"
    )
else:
    model = MusicgenForConditionalGeneration.from_pretrained(model_id)
    model.to(device)

processor = AutoProcessor.from_pretrained(model_id)
SAMPLING_RATE = model.config.audio_encoder.sampling_rate # 32000
TOKENS_PER_SECOND = 50

# Creiamo un Lock globale per impedire l'accesso simultaneo al modello
modello_lock = threading.Lock()

print("MUSIC GEN READY")

# Salva l'array numpy come file .wav
def salva_audio(nome_file, audio_data):
    filepath = os.path.join(OUT_DIR, nome_file)
    scipy.io.wavfile.write(filepath, rate=SAMPLING_RATE, data=audio_data)
    print(f"    [SALVATAGGIO] File aggiornato: {filepath}")

def unisci_tracce(traccia_precedente, traccia_nuova, campioni_sovrapposti):
    if campioni_sovrapposti > 0:
        traccia_senza_coda = traccia_precedente[:-campioni_sovrapposti]
    else:
        traccia_senza_coda = traccia_precedente
    return np.concatenate((traccia_senza_coda, traccia_nuova))

# Genera un brano da zero basato solo sul testo.
def genera_brano_iniziale(prompt, durata_secondi=30):
    print(f"  [GEN START] Richiesta generazione da zero: '{prompt}' ({durata_secondi}s)")
    tokens = int(durata_secondi * TOKENS_PER_SECOND)

    inputs = processor(
        text=[prompt],
        padding=True,
        return_tensors="pt",
    ).to(device)

    # SC: Solo un thread alla volta può usare il modello ---
    with modello_lock:
        print(f"  [GEN START] Lock acquisito. Inizio calcolo per '{prompt}'...")
        # Usa guidance 3.0 per la generazione iniziale (migliore qualità audio)
        audio_values = model.generate(**inputs, max_new_tokens=tokens, do_sample=True, guidance_scale=3.0)

    audio_data = audio_values[0, 0].cpu().numpy().astype("float32")
    return audio_data

# Genera un brano usando gli ultimi secondi del brano precedente.
def genera_continuazione(prompt, audio_precedente, durata_secondi=22, secondi_contesto=8, guidance=2.0):
    print(f"  [GEN CONT] Generazione continuazione: '{prompt}' (richiesti {durata_secondi}s, contesto: {secondi_contesto}s)")

    # LIMITE DI MUSICGEN (Max 30s totali / 1500 tokens) ---
    max_tokens_totali = 1500
    tokens_prompt = int(secondi_contesto * TOKENS_PER_SECOND)
    max_new_tokens_ammessi = max_tokens_totali - tokens_prompt

    tokens_richiesti = int(durata_secondi * TOKENS_PER_SECOND)
    tokens_effettivi = min(tokens_richiesti, max_new_tokens_ammessi)

    if tokens_richiesti > max_new_tokens_ammessi:
        print(f"    [ATTENZIONE] Il modello supporta max 30s totali. Con {secondi_contesto}s di contesto, la nuova generazione è limitata in automatico a {max_new_tokens_ammessi / TOKENS_PER_SECOND}s.")

    # Calcoliamo i campioni da tenere per il prompt
    campioni_da_tenere = int(secondi_contesto * SAMPLING_RATE)

    if len(audio_precedente) < campioni_da_tenere:
        audio_prompt = audio_precedente
        campioni_da_tenere = len(audio_precedente) # Adattiamo se la traccia è troppo corta
    else:
        audio_prompt = audio_precedente[-campioni_da_tenere:]

    inputs = processor(
        audio=audio_prompt,
        sampling_rate=SAMPLING_RATE,
        text=[prompt],
        padding=True,
        return_tensors="pt"
    ).to(device)

    # Cast a float16 se siamo su GPU (OTTIMIZZAZIONE DELLE PRESTAZIONI)
    if device == "cuda":
        inputs = {k: v.to(torch.float16) if torch.is_floating_point(v) else v for k, v in inputs.items()}

    with modello_lock:
        print(f"  [GEN CONT] Lock acquisito. Inizio calcolo per '{prompt}'...")
        # Guidance più bassa per le continuazioni previene la distorsione a cascata
        audio_values = model.generate(**inputs, max_new_tokens=tokens_effettivi, do_sample=True, guidance_scale=guidance)

    audio_data = audio_values[0, 0].cpu().numpy().astype("float32")

    return audio_data, campioni_da_tenere


class ParadigmaRunning:
    def __init__(self, prompt_iniziale="Fast electronic running beat, high energy, 130 bpm",
                 audio_iniziale=None, session_id: str = "", categoria: str = ""):
        print(f"\n=== Inizializzazione Paradigma RUNNING ===")
        self.prompt_base = prompt_iniziale
        self.contatore_file = 1

        if audio_iniziale is not None:
            # Cache hit: usa l'audio fornito, non genera nulla
            print(f"  [CACHE HIT] Audio iniziale fornito dall'esterno, skip generazione.")
            self.ultimo_audio = audio_iniziale
        else:
            # Cache miss: genera normalmente
            self.ultimo_audio = genera_brano_iniziale(self.prompt_base, 30)
            prefix = f"{session_id}_" if session_id else ""
            salva_audio(f"{prefix}{categoria}_chunk_{self.contatore_file:02d}.wav", self.ultimo_audio)
            self.contatore_file += 1

    def aggiorna(self, session_id: str = "", categoria: str = "", newStyle: bool = False):
        print(f"\n[RUNNING] Generazione nuovo segmento (Chunk {self.contatore_file})")

        if newStyle:
            audio_grezzo, campioni_prompt = genera_brano_iniziale(self.prompt_base, 30)
        else :
            audio_grezzo, campioni_prompt = genera_continuazione(
                prompt=self.prompt_base,
                audio_precedente=self.ultimo_audio,
                durata_secondi=22,
                secondi_contesto=8,
                guidance=2.0
            )

        solo_nuovo_audio = audio_grezzo[campioni_prompt:]

        prefix = f"{session_id}_" if session_id else ""
        nome_file = f"{prefix}{categoria}_chunk_{self.contatore_file:02d}.wav"
        salva_audio(nome_file, solo_nuovo_audio)

        self.ultimo_audio = audio_grezzo
        self.contatore_file += 1
        return nome_file

class ParadigmaWeightlifting:
    def __init__(self, prompt_resting="Calm lo-fi beats for resting",
                 prompt_lifting="Heavy metal or trap beat for lifting",
                 audio_iniziale_calmo=None, audio_iniziale_ritmato=None, session_id: str = "", categoria: str = ""):
        print(f"\n=== Inizializzazione Paradigma WEIGHTLIFTING ===")
        self.prompt_calmo = prompt_resting
        self.prompt_ritmato = prompt_lifting
        self.contatore_calmo = 1
        self.contatore_ritmato = 1

        if audio_iniziale_calmo is not None:
            print(f"  [CACHE HIT] Audio calmo fornito dall'esterno, skip generazione.")
            self.ultimo_calmo = audio_iniziale_calmo
        else:
            self.ultimo_calmo = genera_brano_iniziale(self.prompt_calmo, 30)
            prefix = f"{session_id}_" if session_id else ""
            salva_audio(f"{prefix}{categoria}_calmo_chunk_{self.contatore_calmo:02d}.wav", self.ultimo_calmo)
            self.contatore_calmo += 1

        if audio_iniziale_ritmato is not None:
            print(f"  [CACHE HIT] Audio ritmato fornito dall'esterno, skip generazione.")
            self.ultimo_ritmato = audio_iniziale_ritmato
        else:
            self.ultimo_ritmato, _ = genera_continuazione(
                prompt=self.prompt_ritmato,
                audio_precedente=self.ultimo_calmo,
                durata_secondi=25,
                secondi_contesto=5,
                guidance=3.0,
            )
            prefix = f"{session_id}_" if session_id else ""
            salva_audio(f"{prefix}{categoria}_ritmato_chunk_{self.contatore_ritmato:02d}.wav", self.ultimo_ritmato)
            self.contatore_ritmato += 1

    def aggiorna(self, prompt_resting=None, prompt_lifting=None, session_id: str = "", categoria: str = "", newStyle: bool = False):
        print(f"\n[WEIGHTLIFTING] Generazione nuovi segmenti")


        if prompt_resting is not None:
            self.prompt_calmo = prompt_resting

        if prompt_lifting is not None:
            self.prompt_ritmato = prompt_lifting

        prefix = f"{session_id}_" if session_id else ""

        if newStyle:
            solo_nuovo_calmo = genera_brano_iniziale(self.prompt_calmo, 30)

        else:
            audio_grezzo_calmo, prompt_calmo_len = genera_continuazione(
                prompt=self.prompt_calmo,
                audio_precedente=self.ultimo_calmo,
                durata_secondi=22,
                secondi_contesto=8,
                guidance=2.0
            )
            solo_nuovo_calmo = audio_grezzo_calmo[prompt_calmo_len:]
        nome_calmo = f"{prefix}{categoria}_calmo_chunk_{self.contatore_calmo:02d}.wav"
        salva_audio(nome_calmo, solo_nuovo_calmo)
        self.ultimo_calmo = solo_nuovo_calmo
        self.contatore_calmo += 1

        # Aggiornamento Ritmato
        if newStyle:
            solo_nuovo_ritmato = genera_continuazione(
                prompt=self.prompt_ritmato,
                audio_precedente=self.ultimo_calmo,
                durata_secondi=25,
                secondi_contesto=5,
                guidance=3.0,
            )
        else:
            audio_grezzo_ritmato, prompt_ritmato_len = genera_continuazione(
                prompt=self.prompt_ritmato,
                audio_precedente=self.ultimo_ritmato,
                durata_secondi=22,
                secondi_contesto=8,
                guidance=2.0
            )
            solo_nuovo_ritmato = audio_grezzo_ritmato[prompt_ritmato_len:]
        nome_ritmato = f"{prefix}{categoria}_ritmato_chunk_{self.contatore_ritmato:02d}.wav"
        salva_audio(nome_ritmato, solo_nuovo_ritmato)
        self.ultimo_ritmato = solo_nuovo_ritmato
        self.contatore_ritmato += 1

        # Restituisce entrambi i nomi così il server sa esattamente quali cercare
        return nome_calmo, nome_ritmato

class ParadigmaYoga:
    def __init__(self, prompt_iniziale="Ambient drone zen, very slow, relaxing",
                 audio_iniziale=None, session_id: str = "", categoria: str = ""):
        print(f"\n=== Inizializzazione Paradigma YOGA ===")
        self.prompt_base = prompt_iniziale
        self.contatore_file = 1

        if audio_iniziale is not None:
            print(f"  [CACHE HIT] Audio iniziale fornito dall'esterno, skip generazione.")
            self.ultimo_audio = audio_iniziale
        else:
            self.ultimo_audio = genera_brano_iniziale(self.prompt_base, 30)
            prefix = f"{session_id}_" if session_id else ""
            salva_audio(f"{prefix}{categoria}_chunk_{self.contatore_file:02d}.wav", self.ultimo_audio)
            self.contatore_file += 1

    def aggiorna(self, session_id: str = "", categoria: str = "", newStyle: bool = False):
        print(f"\n[YOGA] Generazione nuovo segmento (Chunk {self.contatore_file})")

        if newStyle:
            solo_nuovo_audio = genera_brano_iniziale(self.prompt_base, 30)
        else:
            audio_grezzo, campioni_prompt = genera_continuazione(
                prompt=self.prompt_base,
                audio_precedente=self.ultimo_audio,
                durata_secondi=22,
                secondi_contesto=8,
                guidance=2.0
            )
            solo_nuovo_audio = audio_grezzo[campioni_prompt:]

        # Nome file con prefisso session_id se fornito
        prefix = f"{session_id}_" if session_id else ""
        nome_file = f"{prefix}{categoria}_chunk_{self.contatore_file:02d}.wav"
        salva_audio(nome_file, solo_nuovo_audio)

        self.ultimo_audio = solo_nuovo_audio
        self.contatore_file += 1
        return nome_file

# ESEMPIO DI UTILIZZO
def avvia_simulazione():
    print("\nSimulazione di avvio allenamenti...")

    # Scegliamo un paradigma
    paradigma = ParadigmaYoga()

    print("\n--- INIZIO ALLENAMENTO (Simulazione) ---")

    print("Primo aggiornamento (chiamato manualmente)...")
    paradigma.aggiorna()
    time.sleep(1)

    print("Secondo aggiornamento (chiamato manualmente)...")
    paradigma.aggiorna()

    paradigma.aggiorna()
    paradigma.aggiorna()
    paradigma.aggiorna()
    paradigma.aggiorna()
    paradigma.aggiorna()

    print(f"\nGenerazione completata! Controlla la cartella '{OUT_DIR}' per i file audio.")

if __name__ == "__main__":
    avvia_simulazione()