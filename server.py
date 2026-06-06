"""
  POST /session/start          → Crea una nuova sessione e avvia la prima generazione
  POST /session/update         → Aggiorna il prompt di una sessione esistente
  GET  /session/{id}/status    → Polling: verifica se l'audio è pronto
  GET  /session/{id}/audio     → Scarica il file audio più recente
  GET  /sessions               → Lista di tutte le sessioni attive (debug)
  DELETE /session/{id}         → Chiude e pulisce una sessione
"""
import io
import re
import shutil
import uuid
import time
import threading
import logging
import zipfile
from enum import Enum
from pathlib import Path
from typing import Optional

import scipy
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse

import llm_gen
import music_gen as gen

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("bioResponsiveSoundscape")

app = FastAPI(
    title="BioResponsive Music Generator",
    description="Server di generazione musicale adattiva per allenamenti.",
    version="3.0.0",
)

OUTPUT_DIR = Path("output_allenamento")
OUTPUT_DIR.mkdir(exist_ok=True)

CACHE_DIR = Path("cache_audio")
CACHE_DIR.mkdir(exist_ok=True)

CACHE_MAX_MB = 500


class AudioCache:
    """
    Cache condivisa tra sessioni organizzata per categoria.

    Struttura su disco:
        cache_audio/
            {categoria}/
                chunk_001.wav
                chunk_002.wav
                ...

    Struttura in memoria:
        _index = {
            "cyberpunk_synthwave": [Path(...), Path(...), ...],
            "dark_trap_ambient":   [Path(...), ...],
        }

    Il lock serializza tutte le operazioni per evitare race condition
    tra sessioni concorrenti che leggono o scrivono la stessa categoria.
    """

    def __init__(self, max_mb: float = CACHE_MAX_MB):
        self._index: dict[str, list[Path]] = {}
        self._lock = threading.Lock()
        self._max_bytes = max_mb * 1024 * 1024
        self._rebuild_index()

    def _rebuild_index(self):
        """Ricostruisce l'indice in memoria scansionando la directory cache al boot."""
        for cat_dir in sorted(CACHE_DIR.iterdir()):
            if cat_dir.is_dir():
                chunks = sorted(cat_dir.glob("chunk_*.wav"))
                if chunks:
                    self._index[cat_dir.name] = chunks
                    log.info(f"[CACHE] Categoria '{cat_dir.name}' caricata: {len(chunks)} chunk")

    def _total_bytes(self) -> int:
        total = 0
        for chunks in self._index.values():
            for path in chunks:
                if path.exists():
                    total += path.stat().st_size
        return total

    def _evict_oldest(self):
        """
        Rimuove il chunk più vecchio in assoluto (globalmente tra tutte le categorie)
        finché non si rientra nella soglia. Chiamato sempre dentro il lock.
        """
        while self._total_bytes() > self._max_bytes:
            # Raccoglie tutti i chunk con il loro mtime
            all_chunks = [
                (path, path.stat().st_mtime)
                for chunks in self._index.values()
                for path in chunks
                if path.exists()
            ]
            if not all_chunks:
                break

            # Elimina il più vecchio
            oldest_path, _ = min(all_chunks, key=lambda x: x[1])
            oldest_path.unlink(missing_ok=True)
            log.info(f"[CACHE] Eviction: rimosso {oldest_path.name} (soglia MB superata)")

            # Aggiorna l'indice
            for cat, chunks in list(self._index.items()):
                if oldest_path in chunks:
                    chunks.remove(oldest_path)
                    if not chunks:
                        del self._index[cat]
                    break

    def get(self, categoria: str) -> Optional[list[Path]]:
        """
        Restituisce la lista di chunk disponibili per la categoria, o None se assente.
        Verifica che i file esistano fisicamente sul disco.
        """
        with self._lock:
            chunks = self._index.get(categoria)
            if not chunks:
                return None

            # Filtra solo i file che esistono davvero nel file system
            valid_chunks = [p for p in chunks if p.exists()]

            return valid_chunks if valid_chunks else None

    def add(self, categoria: str, src_path1: Path, src_path2: Path = None) -> Path:
        with self._lock:
            cat_dir = CACHE_DIR / categoria
            cat_dir.mkdir(exist_ok=True)

            existing = self._index.get(categoria, [])

            if src_path2:
                # chunk_idx basato su coppie: ogni coppia conta come 1
                chunk_idx = (len(existing) // 2) + 1
                dst1 = cat_dir / f"{categoria}_chunk_lifting_{chunk_idx:03d}.wav"
                dst2 = cat_dir / f"{categoria}_chunk_resting_{chunk_idx:03d}.wav"
                shutil.copy2(src_path1, dst1)
                shutil.copy2(src_path2, dst2)  # era src_path1 — bug
                log.info(f"[CACHE] Aggiunti '{dst1.name}' e '{dst2.name}' in '{categoria}'")
                self._index.setdefault(categoria, []).extend([dst1, dst2])
                self._evict_oldest()
                return dst1
            else:
                chunk_idx = len(existing) + 1
                dst = cat_dir / f"{categoria}_chunk_{chunk_idx:03d}.wav"
                shutil.copy2(src_path1, dst)
                log.info(f"[CACHE] Aggiunto '{dst.name}' in '{categoria}'")
                self._index.setdefault(categoria, []).append(dst)
                self._evict_oldest()
                return dst

    def stats(self) -> dict:
        """Restituisce statistiche sulla cache (usato dall'endpoint /cache/stats)."""
        with self._lock:
            return {
                "total_mb": round(self._total_bytes() / (1024 * 1024), 2),
                "max_mb": CACHE_MAX_MB,
                "categories": {
                    cat: len(chunks)
                    for cat, chunks in self._index.items()
                },
            }


# Istanza globale della cache
audio_cache = AudioCache()


# Payload JSON
class RunningPayload(BaseModel):
    activity: str = "running"
    user_initial_intent: str
    current_avg_hr: int
    cadence_spm: int
    performance_trend: str          # "rallentamento" | "stabile" | "accelerazione"
    target_bpm: int

class WeightliftingPayload(BaseModel):
    activity: str = "weightlifting"
    user_initial_intent: str
    last_hr_peak: int
    average_hr: int
    time_in_current_state_sec: int

class YogaPayload(BaseModel):
    activity: str = "yoga_cooldown"
    user_initial_intent: str
    current_avg_hr: int
    hr_trend: str                   # "stabile" | "in calo" | "in aumento"
    target_goal: str

ActivityPayload = RunningPayload | WeightliftingPayload | YogaPayload

class StartRequest(BaseModel):
    payload: dict

class UpdateRequest(BaseModel):
    payload: dict


class SessionStatus(str, Enum):
    GENERATING = "generating"
    READY      = "ready"
    ERROR      = "error"

class Session:
    def __init__(self, session_id: str, activity: str):
        self.session_id: str              = session_id
        self.activity: str                = activity
        self.status: SessionStatus        = SessionStatus.GENERATING
        self.paradigma                    = None
        self.categoria                    = None
        self.latest_audio_path: Optional[Path] = None
        self.latest_audio_path2: Optional[Path] = None
        self.error_message: Optional[str] = None
        self.created_at: float            = time.time()
        self.updated_at: float            = time.time()
        # protegge lo stato interno di una singola sessione
        self._lock: threading.Lock        = threading.Lock()

    def set_ready(self, path1: Path, path2: Path = None):
        with self._lock:
            self.latest_audio_path = path1
            self.latest_audio_path2 = path2  # azzera esplicitamente anche se None
            self.status = SessionStatus.READY
            self.updated_at = time.time()

    def set_generating(self):
        with self._lock:
            self.status = SessionStatus.GENERATING
            self.updated_at = time.time()

    def set_error(self, msg: str):
        with self._lock:
            self.status = SessionStatus.ERROR
            self.error_message = msg
            self.updated_at = time.time()


_sessions: dict[str, Session] = {}
# Protegge il dizionario _sessions dall'accesso concorrente di più thread generati da FastAPI
_sessions_lock = threading.Lock()

# Utility
def _get_session(session_id: str) -> Session:
    with _sessions_lock:
        session = _sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Sessione '{session_id}' non trovata.")
    return session


def _find_latest_wav(session_id: str, prefix: str) -> Optional[Path]:
    """Trova il file .wav più recente per questa sessione."""
    candidates = sorted(OUTPUT_DIR.glob(f"{prefix}*.wav"))
    return candidates[-1] if candidates else None

def _copy_for_session(session_id: str, original_name: str) -> Path:
    src = OUTPUT_DIR / original_name
    dst = OUTPUT_DIR / f"{session_id}_{original_name}"

    if src.exists():
        shutil.copy2(src, dst)
        src.unlink()
        return dst
    elif dst.exists():
        return dst
    else:
        raise FileNotFoundError(f"File audio generato non trovato: {src}")


# INIZIALIZZAZIONE in background
def _bg_init_running(session: Session, payload: dict):
    try:
        log.info(f"[{session.session_id}] Generazione prompt Running via LLM...")
        categoria, musicgen_prompt = llm_gen.genera_prompt_musicgen(payload)
        session.categoria = categoria
        log.info(f"[{session.session_id}] Categoria: '{categoria}' | Prompt: {musicgen_prompt}")

        # Controlla se esiste già audio in cache per questa categoria
        cached_chunks = audio_cache.get(categoria)
        if cached_chunks:
            audio_iniziale = scipy.io.wavfile.read(cached_chunks[0])[1].astype("float32")
            paradigma = gen.ParadigmaRunning(
                prompt_iniziale=musicgen_prompt,
                audio_iniziale=audio_iniziale,
                session_id=session.session_id,
                categoria=categoria,
            )
            session.paradigma = paradigma
            session.set_ready(cached_chunks[0])
        else:
            log.info(f"[{session.session_id}] Cache MISS per '{categoria}': avvio generazione")
            paradigma = gen.ParadigmaRunning(prompt_iniziale=musicgen_prompt, categoria=categoria)
            session.paradigma = paradigma
            audio_path = _copy_for_session(session.session_id, f"{categoria}_chunk_01.wav")
            # Salva il chunk generato in cache per sessioni future
            audio_cache.add(categoria, audio_path)
            session.set_ready(audio_path)

        log.info(f"[{session.session_id}] Inizializzazione Running completata → {session.latest_audio_path}")

    except Exception as e:
        log.exception(f"[{session.session_id}] Errore init Running")
        session.set_error(str(e))


def _bg_init_weightlifting(session: Session, payload: dict):
    try:
        log.info(f"[{session.session_id}] Generazione prompt Weightlifting via LLM...")
        categoria, lifting_prompt, recovery_prompt = llm_gen.genera_prompt_musicgen(payload)
        session.categoria = categoria
        log.info(
            f"[{session.session_id}] Categoria: '{categoria}' | Lifting: {lifting_prompt} | Recovery: {recovery_prompt}")

        cached_chunks = audio_cache.get(categoria)
        if cached_chunks:
            path_calmo_cache = next((p for p in cached_chunks if "resting_01" in p.name), None)
            path_ritmato_cache = next((p for p in cached_chunks if "lifting_01" in p.name), None)

            if path_calmo_cache:
                audio_calmo = scipy.io.wavfile.read(path_calmo_cache)[1].astype("float32")
            else:
                audio_calmo = None
            if path_ritmato_cache:
                audio_ritmato = scipy.io.wavfile.read(path_ritmato_cache)[1].astype("float32")
            else:
                audio_ritmato = None

            paradigma = gen.ParadigmaWeightlifting(
                prompt_resting=recovery_prompt,
                prompt_lifting=lifting_prompt,
                audio_iniziale_calmo=audio_calmo,
                audio_iniziale_ritmato=audio_ritmato,
                session_id=session.session_id,
                categoria=session.categoria,
            )
            session.paradigma = paradigma
            session.set_ready(
                path_ritmato_cache or cached_chunks[0],
                path_calmo_cache,
            )
        else:
            log.info(f"[{session.session_id}] Cache MISS per '{categoria}': avvio generazione")
            paradigma = gen.ParadigmaWeightlifting(
                prompt_resting=recovery_prompt,
                prompt_lifting=lifting_prompt,
                categoria=session.categoria,
            )
            session.paradigma = paradigma

            wav_name1 = f"{session.categoria}_ritmato_chunk_01.wav"
            wav_name2 = f"{session.categoria}_calmo_chunk_01.wav"

            audio_path1 = _copy_for_session(session.session_id, wav_name1)
            audio_path2 = _copy_for_session(session.session_id, wav_name2)
            audio_cache.add(categoria, audio_path1, audio_path2)
            session.set_ready(audio_path1, audio_path2)

        log.info(f"[{session.session_id}] Inizializzazione Weightlifting completata → {session.latest_audio_path}")

    except Exception as e:
        log.exception(f"[{session.session_id}] Errore init Weightlifting")
        session.set_error(str(e))


def _bg_init_yoga(session: Session, payload: dict):
    try:
        log.info(f"[{session.session_id}] Generazione prompt Yoga via LLM...")
        categoria, musicgen_prompt = llm_gen.genera_prompt_musicgen(payload)
        session.categoria = categoria
        log.info(f"[{session.session_id}] Categoria: '{categoria}' | Prompt: {musicgen_prompt}")

        cached_chunks = audio_cache.get(categoria)
        if cached_chunks:
            audio_iniziale = scipy.io.wavfile.read(cached_chunks[0])[1].astype("float32")
            paradigma = gen.ParadigmaYoga(
                prompt_iniziale=musicgen_prompt,
                audio_iniziale=audio_iniziale,
                session_id=session.session_id,
                categoria=session.categoria,
            )
            session.paradigma = paradigma
            session.set_ready(cached_chunks[0])
        else:
            log.info(f"[{session.session_id}] Cache MISS per '{categoria}': avvio generazione")
            paradigma = gen.ParadigmaYoga(prompt_iniziale=musicgen_prompt, categoria=session.categoria)
            session.paradigma = paradigma
            audio_path = _copy_for_session(session.session_id, f"{categoria}_chunk_01.wav")
            audio_cache.add(categoria, audio_path)
            session.set_ready(audio_path)

        log.info(f"[{session.session_id}] Inizializzazione Yoga completata → {session.latest_audio_path}")

    except Exception as e:
        log.exception(f"[{session.session_id}] Errore init Yoga")
        session.set_error(str(e))


# AGGIORNAMENTO in background
def _bg_update_running(session: Session, payload: dict):
    try:
        session.set_generating()
        log.info(f"[{session.session_id}] Aggiornamento Running...")

        old_path = session.latest_audio_path

        risultato_llm = llm_gen.genera_prompt_musicgen(payload)
        if isinstance(risultato_llm, tuple):
            categoria, new_prompt = risultato_llm
        else:
            new_prompt = risultato_llm
            categoria = session.categoria

        categoria_cambiata = categoria != session.categoria
        session.categoria = categoria
        session.paradigma.prompt_base = new_prompt
        log.info(f"[{session.session_id}] Categoria: '{categoria}' | Nuovo prompt: {new_prompt}")

        if categoria_cambiata:
            log.info(f"[{session.session_id}] Cambio categoria: '{categoria}' — reset contesto audio")
            cached_chunks = audio_cache.get(categoria)

            if cached_chunks:
                audio_iniziale = scipy.io.wavfile.read(cached_chunks[0])[1].astype("float32")
                session.paradigma.ultimo_audio = audio_iniziale
                session.paradigma.contatore_file = 0
                session_audio_path = OUTPUT_DIR / f"{session.session_id}_running_{categoria_cambiata}_chunk_01.wav"
                shutil.copy2(cached_chunks[0], session_audio_path)
                session.set_ready(session_audio_path)
            else:
                nome_file = session.paradigma.aggiorna(session_id=session.session_id, categoria=categoria_cambiata)
                audio_path = OUTPUT_DIR / nome_file
                audio_cache.add(categoria, audio_path)
                session.paradigma.contatore_file = 0
                session.set_ready(audio_path)

            if old_path and old_path.exists() and CACHE_DIR not in old_path.parents:
                old_path.unlink(missing_ok=True)
            return

        # Stessa categoria — logica cache hit/miss invariata
        target_chunk_number = session.paradigma.contatore_file + 1
        target_filename = f"chunk_{target_chunk_number:03d}.wav"
        cached_chunks = audio_cache.get(categoria)
        target_path_in_cache = None

        # Ricerca per nome file e non per indice
        if cached_chunks:
            for path in cached_chunks:
                if path.name == target_filename:
                    target_path_in_cache = path
                    break

        if target_path_in_cache and target_path_in_cache.exists():
            # --- CACHE HIT SICURO ---
            log.info(f"[{session.session_id}] Cache HIT per '{categoria}': trovato {target_filename}")

            chunk_idx = session.paradigma.contatore_file
            wav_name = f"running_chunk_{chunk_idx:02d}.wav"

            # Creiamo una copia isolata per la sessione, immune all'eviction della cache
            session_audio_path = OUTPUT_DIR / f"{session.session_id}_{wav_name}"
            shutil.copy2(target_path_in_cache, session_audio_path)

            session.paradigma.contatore_file += 1
            session.set_ready(session_audio_path)


        else:
            log.info(f"[{session.session_id}] Cache MISS per '{categoria}': avvio generazione")
            nome_file = session.paradigma.aggiorna(session_id=session.session_id, categoria=session.categoria)
            audio_path = OUTPUT_DIR / nome_file
            audio_cache.add(categoria, audio_path)
            session.set_ready(audio_path)

        # 3. Pulizia sicura del chunk di sessione precedente
        if old_path and old_path.exists() and CACHE_DIR not in old_path.parents:
            old_path.unlink(missing_ok=True)

    except Exception as e:
        log.exception(f"[{session.session_id}] Errore update Running")
        session.set_error(str(e))


def _bg_update_weightlifting(session: Session, payload: dict):
    try:
        session.set_generating()
        log.info(f"[{session.session_id}] Aggiornamento Weightlifting...")

        old_path1 = session.latest_audio_path
        old_path2 = session.latest_audio_path2

        # Estrazione e parsing del prompt LLM
        risultato_llm = llm_gen.genera_prompt_musicgen(payload)
        if len(risultato_llm) == 3:
            categoria, lifting_prompt, recovery_prompt = risultato_llm
        else:
            lifting_prompt, recovery_prompt = risultato_llm
            categoria = getattr(session, "categoria", "default")

        categoria_precedente = getattr(session, "categoria", None)
        categoria_cambiata = (categoria_precedente is not None) and (categoria != categoria_precedente)
        session.categoria = categoria

        log.info(
            f"[{session.session_id}] Categoria: '{categoria}' | Lifting: {lifting_prompt} | Recovery: {recovery_prompt}")

        # Poiché generiamo e serviamo entrambe le tracce in parallelo, usiamo un solo contatore come riferimento
        chunk_idx = session.paradigma.contatore_ritmato

        def genera_e_salva():
            nome_calmo, nome_ritmato = session.paradigma.aggiorna(
                prompt_lifting=lifting_prompt,
                prompt_resting=recovery_prompt,
                session_id=session.session_id,
                categoria=session.categoria,
            )
            path_calmo = OUTPUT_DIR / nome_calmo
            path_ritmato = OUTPUT_DIR / nome_ritmato

            audio_cache.add(categoria, path_ritmato, path_calmo)

            # Passiamo entrambe le tracce al nuovo metodo set_ready
            session.set_ready(path_ritmato, path_calmo)

        # Cambio categoria
        if categoria_cambiata:
            log.info(f"[{session.session_id}] Cambio categoria: '{categoria}' — reset contesto audio")
            cached_chunks = audio_cache.get(categoria) or []

            path_calmo_cache = next((p for p in cached_chunks if "resting_01" in p.name), None)
            path_ritmato_cache = next((p for p in cached_chunks if "lifting_01" in p.name), None)

            if path_calmo_cache and path_ritmato_cache:
                audio_calmo = scipy.io.wavfile.read(path_calmo_cache)[1].astype("float32")
                audio_ritmato = scipy.io.wavfile.read(path_ritmato_cache)[1].astype("float32")

                session.paradigma.ultimo_calmo = audio_calmo
                session.paradigma.ultimo_ritmato = audio_ritmato
                session.paradigma.contatore_calmo = 1
                session.paradigma.contatore_ritmato = 1

                session_path_calmo = OUTPUT_DIR / f"{session.session_id}_{path_calmo_cache.name}"
                session_path_ritmato = OUTPUT_DIR / f"{session.session_id}_{path_ritmato_cache.name}"

                shutil.copy2(path_calmo_cache, session_path_calmo)
                shutil.copy2(path_ritmato_cache, session_path_ritmato)

                session.set_ready(session_path_ritmato, session_path_calmo)
            else:
                # Categoria non in cache o chunk 01 mancante: genera da zero
                session.paradigma.contatore_calmo = 1
                session.paradigma.contatore_ritmato = 1
                genera_e_salva()

        # 4. Logica a parità di categoria
        else:
            target_calmo = f"weightlifting_resting_{session.categoria}_{chunk_idx:02d}.wav"
            target_ritmato = f"weightlifting_lifting_{session.categoria}_{chunk_idx:02d}.wav"

            cached_chunks = audio_cache.get(categoria) or []
            path_calmo_cache = next((p for p in cached_chunks if target_calmo in p.name), None)
            path_ritmato_cache = next((p for p in cached_chunks if target_ritmato in p.name), None)

            if path_calmo_cache and path_calmo_cache.exists() and path_ritmato_cache and path_ritmato_cache.exists():
                # --- CACHE HIT SICURO ---
                log.info(f"[{session.session_id}] Cache HIT per '{categoria}': trovata coppia {chunk_idx:02d}")

                session_path_calmo = OUTPUT_DIR / f"{session.session_id}_{target_calmo}"
                session_path_ritmato = OUTPUT_DIR / f"{session.session_id}_{target_ritmato}"

                shutil.copy2(path_calmo_cache, session_path_calmo)
                shutil.copy2(path_ritmato_cache, session_path_ritmato)

                session.paradigma.contatore_calmo += 1
                session.paradigma.contatore_ritmato += 1

                session.set_ready(session_path_ritmato, session_path_calmo)
            else:
                # --- CACHE MISS ---
                log.info(
                    f"[{session.session_id}] Cache MISS per '{categoria}': avvio generazione coppia {chunk_idx:02d}")
                genera_e_salva()

        # 5. Pulizia dei vecchi file
        for old_path in (old_path1, old_path2):
            if old_path and old_path.exists() and CACHE_DIR not in old_path.parents:
                old_path.unlink(missing_ok=True)

    except Exception as e:
        log.exception(f"[{session.session_id}] Errore update Weightlifting")
        session.set_error(str(e))


def _bg_update_yoga(session: Session, payload: dict):
    try:
        session.set_generating()
        log.info(f"[{session.session_id}] Aggiornamento Yoga...")

        old_path = session.latest_audio_path

        risultato_llm = llm_gen.genera_prompt_musicgen(payload)
        if isinstance(risultato_llm, tuple):
            categoria, new_prompt = risultato_llm
        else:
            new_prompt = risultato_llm
            categoria = session.categoria

        # Calcola il cambio PRIMA di aggiornare session.categoria
        categoria_cambiata = categoria != session.categoria
        session.categoria = categoria
        session.paradigma.prompt_base = new_prompt

        if categoria_cambiata:
            log.info(f"[{session.session_id}] Cambio categoria: '{categoria_cambiata}' — reset contesto audio")
            cached_chunks = audio_cache.get(categoria)

            if cached_chunks:
                audio_iniziale = scipy.io.wavfile.read(cached_chunks[0])[1].astype("float32")
                session.paradigma.ultimo_audio = audio_iniziale
                session.paradigma.contatore_file = 0
                session_audio_path = OUTPUT_DIR / f"{session.session_id}_yoga_{categoria_cambiata}_chunk_01.wav"
                shutil.copy2(cached_chunks[0], session_audio_path)
                session.set_ready(session_audio_path)
            else:
                # Nuova categoria mai vista: genera da zero con il nuovo prompt
                nome_file = session.paradigma.aggiorna(session_id=session.session_id, categoria=categoria_cambiata)
                audio_path = OUTPUT_DIR / nome_file
                audio_cache.add(categoria, audio_path)
                session.paradigma.contatore_file = 0
                session.set_ready(audio_path)

            if old_path and old_path.exists() and CACHE_DIR not in old_path.parents:
                old_path.unlink(missing_ok=True)
            return  # esce prima della logica normale

        # Calcolo del nome esatto del chunk
        target_chunk_number = session.paradigma.contatore_file + 1
        target_filename = f"chunk_{target_chunk_number:03d}.wav"

        cached_chunks = audio_cache.get(categoria)
        target_path_in_cache = None

        if cached_chunks:
            for path in cached_chunks:
                if path.name == target_filename:
                    target_path_in_cache = path
                    break

        if target_path_in_cache and target_path_in_cache.exists():
            # --- CACHE HIT SICURO ---
            log.info(f"[{session.session_id}] Cache HIT per '{categoria}': trovato {target_filename}")

            chunk_idx = session.paradigma.contatore_file
            wav_name = f"yoga_{categoria}_chunk_{chunk_idx:02d}.wav"

            session_audio_path = OUTPUT_DIR / f"{session.session_id}_{wav_name}"
            shutil.copy2(target_path_in_cache, session_audio_path)

            session.paradigma.contatore_file += 1
            session.set_ready(session_audio_path)


        else:
            # --- CACHE MISS ---
            log.info(f"[{session.session_id}] Cache MISS per '{categoria}': avvio generazione")
            nome_file = session.paradigma.aggiorna(session_id=session.session_id, categoria=session.categoria)
            audio_path = OUTPUT_DIR / nome_file
            audio_cache.add(categoria, audio_path)
            session.set_ready(audio_path)

        if old_path and old_path.exists() and CACHE_DIR not in old_path.parents:
            old_path.unlink(missing_ok=True)

    except Exception as e:
        log.exception(f"[{session.session_id}] Errore update Yoga")
        session.set_error(str(e))

# Gestione file: ogni sessione ha i propri chunk rinominati con session_id
# Costruisce il path del file audio
def _session_audio_path(session_id: str, original_name: str) -> Path:
    return OUTPUT_DIR / f"{session_id}_{original_name}"


def _rename_latest_for_session(session_id: str, original_name: str) -> Path:
    """
    Sposta/rinomina il file generato da music_gen aggiungendo il session_id come prefisso.
    """
    src = OUTPUT_DIR / original_name
    dst = OUTPUT_DIR / f"{session_id}_{original_name}"

    if src.exists():
        src.rename(dst)
        return dst
    elif dst.exists():
        return dst
    else:
        raise FileNotFoundError(f"File audio non trovato: {src}")


# Dispatcher
_INIT_HANDLERS = {
    "running":        _bg_init_running,
    "weightlifting":  _bg_init_weightlifting,
    "yoga_cooldown":  _bg_init_yoga,
}

_UPDATE_HANDLERS = {
    "running":        _bg_update_running,
    "weightlifting":  _bg_update_weightlifting,
    "yoga_cooldown":  _bg_update_yoga,
}


# Avvia una nuova sessione e restiuisce l'ID
@app.post("/session/start", summary="Avvia una nuova sessione di allenamento")
def start_session(request: StartRequest, background_tasks: BackgroundTasks):
    payload = request.payload
    activity = payload.get("activity")

    if activity not in _INIT_HANDLERS:
        raise HTTPException(
            status_code=400,
            detail=f"Attività '{activity}' non supportata. Usa: {list(_INIT_HANDLERS.keys())}",
        )

    session_id = str(uuid.uuid4())
    session = Session(session_id=session_id, activity=activity)

    #Acquisisce il lock e lo rilascia al termine del with
    with _sessions_lock:
        _sessions[session_id] = session

    log.info(f"Nuova sessione creata: {session_id} (attività: {activity})")

    # Avvia il task di generazione in background (thread separato)
    handler = _INIT_HANDLERS[activity]
    thread = threading.Thread(
        target=handler,
        args=(session, payload),
        daemon=True,
        name=f"init-{session_id[:8]}",
    )
    thread.start()

    return {
        "session_id": session_id,
        "activity": activity,
        "status": SessionStatus.GENERATING,
        "message": "Sessione avviata. Usa GET /session/{session_id}/status per il polling.",
    }


# Aggiorna la sessione con nuovi dati biometrici (409 -> la sessione è già in generazione)
@app.post("/session/{session_id}/update", summary="Aggiorna i dati biometrici e genera nuovo chunk")
def update_session(session_id: str, request: UpdateRequest, background_tasks: BackgroundTasks):
    session = _get_session(session_id)

    if session.status == SessionStatus.READY:
        payload = request.payload
        activity = payload.get("activity", session.activity)

        if activity not in _UPDATE_HANDLERS:
            raise HTTPException(status_code=400, detail=f"Attività '{activity}' non supportata.")

        handler = _UPDATE_HANDLERS[activity]
        thread = threading.Thread(
            target=handler,
            args=(session, payload),
            daemon=True,
            name=f"update-{session_id[:8]}",
        )
        thread.start()
        log.info(f"Aggiornamento sessione {session_id} avviato: {payload}")
        return {
            "session_id": session_id,
            "status": SessionStatus.READY,
            "message": "Aggiornamento avviato. Usa GET /session/{session_id}/status per il polling.",
        }
    else:
        return {
            "session_id": session_id,
            "status": SessionStatus.GENERATING,
            "message": "Aggiornamento avviato. Usa GET /session/{session_id}/status per il polling.",
        }




# Endpoint: Polling dello stato -> Restituisce lo stato corrente della sessione
@app.get("/session/{session_id}/status", summary="Verifica lo stato della generazione (polling)")
def get_status(session_id: str):
    """
    - `generating` → generazione in corso, riprova tra qualche secondo
    - `ready`       → audio pronto, chiama GET /session/{id}/audio
    - `error`       → qualcosa è andato storto
    """
    session = _get_session(session_id)

    response = {
        "session_id": session_id,
        "activity": session.activity,
        "status": session.status,
        "updated_at": session.updated_at,
    }

    if session.status == SessionStatus.READY and session.latest_audio_path:
        response["audio_url"] = f"/session/{session_id}/audio"
        response["audio_filename"] = session.latest_audio_path.name
        response["audio_type"] = "zip" if session.latest_audio_path2 else "wav"

    if session.status == SessionStatus.GENERATING and session.latest_audio_path is not None:
        response["audio_url"] = f"/session/{session_id}/audio"
        response["audio_filename"] = session.latest_audio_path.name
        response["audio_type"] = "zip" if session.latest_audio_path2 else "wav"

    if session.status == SessionStatus.ERROR:
        response["error"] = session.error_message

    return response


# Endpoint: Download del file audio .wav più recente della sessione
@app.get("/session/{session_id}/audio", summary="Scarica l'ultimo file audio generato")
def get_audio(session_id: str):
    session = _get_session(session_id)

    if session.status == SessionStatus.GENERATING and session.latest_audio_path is None:
        raise HTTPException(
            status_code=202,
            detail="Generazione ancora in corso. Riprova quando lo status è 'ready'.",
        )

    if session.status == SessionStatus.ERROR:
        raise HTTPException(
            status_code=500,
            detail=f"La sessione è in stato di errore: {session.error_message}",
        )

    if not session.latest_audio_path or not session.latest_audio_path.exists():
        raise HTTPException(status_code=404, detail="File audio non trovato.")

    # Se esiste anche il secondo file audio, li zippiamo insieme
    if session.latest_audio_path2 and session.latest_audio_path2.exists():
        # Creiamo un buffer in memoria per evitare di scrivere lo zip su disco
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            # Aggiungiamo il primo file
            zip_file.write(session.latest_audio_path, arcname=session.latest_audio_path.name)
            # Aggiungiamo il secondo file
            zip_file.write(session.latest_audio_path2, arcname=session.latest_audio_path2.name)

        # Riportiamo il cursore del buffer all'inizio prima di leggerlo
        zip_buffer.seek(0)

        return StreamingResponse(
            zip_buffer,
            media_type="application/zip",
            headers={
                "Content-Disposition": f"attachment; filename=audio_{session_id}.zip"
            }
        )
    else:
        return FileResponse(
            path=str(session.latest_audio_path),
            media_type="audio/wav",
            filename=session.latest_audio_path.name,
        )


# Endpoint: Lista sessioni (debug / admin)
@app.get("/sessions", summary="Lista tutte le sessioni attive")
def list_sessions():
    with _sessions_lock:
        sessions_snapshot = list(_sessions.values())

    return [
        {
            "session_id": s.session_id,
            "activity": s.activity,
            "status": s.status,
            "created_at": s.created_at,
            "updated_at": s.updated_at,
        }
        for s in sessions_snapshot
    ]


# Endpoint: Elimina una sessione e pulisce i suoi file
@app.delete("/session/{session_id}", summary="Chiude e pulisce una sessione")
def delete_session(session_id: str):
    # Controllo esplicito dell'esistenza
    with _sessions_lock:
        if session_id not in _sessions:
            raise HTTPException(status_code=404, detail=f"Sessione '{session_id}' non trovata.")
        del _sessions[session_id]

    # Pulizia file
    deleted_files = []
    for wav_file in OUTPUT_DIR.glob(f"{session_id}_*.wav"):
        wav_file.unlink(missing_ok=True)
        deleted_files.append(wav_file.name)

    log.info(f"Sessione {session_id} eliminata. File rimossi: {deleted_files}")

    return {
        "session_id": session_id,
        "deleted": True,
        "files_removed": deleted_files,
    }


if __name__ == "__main__":
    import uvicorn
    import threading
    from pyngrok import ngrok

    # Avvia uvicorn in background
    server_thread = threading.Thread(
        target=uvicorn.run,
        kwargs={"app": app, "host": "0.0.0.0", "port": 8000},
        daemon=True,
    )
    server_thread.start()

    # Aspetta che uvicorn sia pronto
    time.sleep(2)

    # Ora apri il tunnel
    ngrok.set_auth_token("3DUXjD1pVA9RanjW3Ud5Um3ix52_87jx7vVT5zAfecbFwKYDg")
    tunnel = ngrok.connect(8000)
    log.info(f"URL pubblico ngrok: {tunnel.public_url}")

    # Mantieni il processo principale vivo
    server_thread.join()

