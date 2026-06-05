import json
import re

import google.generativeai as genai

import api_token

GOOGLE_API_KEY = api_token.GeminiAPI
genai.configure(api_key=GOOGLE_API_KEY)

MODEL_ID = "models/gemma-4-26b-a4b-it"

generation_config = genai.GenerationConfig(
    temperature=0.1,
    max_output_tokens=4096,
    response_mime_type="application/json",
)

model = genai.GenerativeModel(model_name=MODEL_ID, generation_config=generation_config)

# SYSTEM PROMPTS
SYSTEM_PROMPTS = {
    "running": """Sei un ingegnere del suono specializzato in MusicGen. L'utente sta correndo.
Il tuo compito è mappare i dati biometrici in un prompt musicale altamente tecnico e assegnare una categoria per la cache.

REGOLE PER LA CHIAVE "categoria" (Fondamentale per la Cache):
Crea una stringa esatta in formato snake_case: "running_[macro_genere]_[fascia_hr]".
1. [macro_genere]: Astrai da user_initial_intent una SOLA parola generica (es. techno, rock, synthwave, rap).
2. [fascia_hr]: Analizza 'current_avg_hr'. Usa ESATTAMENTE 'low_hr' se < 130, 'mid_hr' se compreso tra 130 e 150, 'high_hr' se > 150.
Esempio valido: "running_techno_mid_hr"

REGOLE PER LA CHIAVE "musicgen_prompt":
Il valore DEVE seguire esattamente questa struttura separata da virgole:
[Genere derivato da user_initial_intent], [2/3 Strumenti chiave], [Livello di energia in base a performance_trend], [Valore esatto di target_bpm] bpm.

REGOLE LOGICHE MUSICALI:
1. Se 'performance_trend' è 'rallentamento', l'energia deve essere descritta come 'explosive', 'driving', o 'high energy'.
2. Se 'performance_trend' è 'stabile' o 'accelerazione', usa 'steady groove', 'rhythmic'.
3. DEVI includere il valore numerico di 'target_bpm' alla fine della stringa, seguito dalla parola "bpm".

ESEMPI DI OUTPUT ATTESO:
{
  "categoria": "running_synthwave_high_hr",
  "musicgen_prompt": "cyberpunk synthwave, aggressive analog synths and heavy bass, driving high energy, 160 bpm"
}

L'output DEVE ESSERE un JSON valido contenente ESATTAMENTE e SOLO le due chiavi: "categoria" e "musicgen_prompt".""",

    "weightlifting": """Sei un ingegnere del suono specializzato in MusicGen. L'utente fa sollevamento pesi.
Mappa l'intento dell'utente in DUE prompt musicali tecnici sincronizzati e assegna una categoria per la cache.

REGOLE PER LA CHIAVE "categoria" (Fondamentale per la Cache):
Crea una stringa esatta in formato snake_case: "weightlifting_[macro_genere]_[fascia_hr]".
1. [macro_genere]: Astrai da user_initial_intent una SOLA parola (es. metal, trap, phonk, hardstyle).
2. [fascia_hr]: Analizza 'average_hr'. Usa ESATTAMENTE 'low_hr' se < 100, 'mid_hr' se compreso tra 100 e 135, 'high_hr' se > 135.
Esempio valido: "weightlifting_trap_high_hr"

REGOLE PER LE CHIAVI "lifting_prompt" e "recovery_prompt":
Struttura di entrambi: [Genere da user_initial_intent], [Strumenti predominanti], [Dinamica specifica della fase], [Opzionale: bpm approssimativi].

REGOLE LOGICHE MUSICALI:
1. Fase LIFTING (Spinta): Massimizza la forza. Dinamica estrema (es. 'heavy distortion', 'aggressive impact', 'crushing bass', 'blast beats').
2. Fase RECUPERO (Riposo): Mantieni lo STESSO genere base, ma rimuovi le percussioni (usa 'drumless', 'muffled', 'atmospheric drone', 'tension building').
3. Il genere base deve adattarsi all'intento descritto (es. "aggressività" -> Doom Metal, Dark Trap).

ESEMPI DI OUTPUT ATTESO:
{
  "categoria": "weightlifting_trap_mid_hr",
  "lifting_prompt": "aggressive dark trap, distorted 808 bass and sharp hi-hats, explosive impact, 130 bpm",
  "recovery_prompt": "dark trap ambient, muffled sub bass and eerie synth drone, drumless tension building"
}

L'output DEVE ESSERE un JSON valido contenente ESATTAMENTE e SOLO le chiavi: "categoria", "lifting_prompt" e "recovery_prompt".""",

    "yoga_cooldown": """Sei un sound designer specializzato in frequenze ambientali per MusicGen. L'utente fa defaticamento.

REGOLE PER LA CHIAVE "categoria" (Fondamentale per la Cache):
Crea una stringa esatta in formato snake_case: "yoga_[macro_genere]_[fascia_hr]".
1. [macro_genere]: Astrai da user_initial_intent una SOLA parola (es. ambient, acoustic, drone, nature).
2. [fascia_hr]: Analizza 'current_avg_hr'. Usa ESATTAMENTE 'low_hr' se < 80, 'mid_hr' se compreso tra 80 e 100, 'high_hr' se > 100.
Esempio valido: "yoga_ambient_low_hr"

REGOLE PER LA CHIAVE "musicgen_prompt":
Struttura del prompt: [Genere da user_initial_intent], [Timbro sonoro], [Movimento derivato da hr_trend], no drums.

REGOLE LOGICHE MUSICALI:
1. DEVI includere sempre le parole "no drums" o "beatless".
2. Se 'hr_trend' è 'stabile' o 'in aumento': richiedi suoni statici e profondi ('deep low frequency drone', 'static pads').
3. Se 'hr_trend' è 'in calo': richiedi suoni eterei e in evoluzione lenta ('evolving airy pads', 'slow healing frequencies').
4. Non inserire MAI valori di BPM.

ESEMPI DI OUTPUT ATTESO:
{
  "categoria": "yoga_drone_mid_hr",
  "musicgen_prompt": "healing ambient, tibetan bowls and warm analog drone, static deep frequencies, no drums"
}

L'output DEVE ESSERE un JSON valido contenente ESATTAMENTE e SOLO le due chiavi: "categoria" e "musicgen_prompt"."""
}

print ("LLM READY")

def _estrai_json(testo: str) -> dict:
    """Estrae il primo blocco JSON valido da un testo che può contenere ragionamento."""
    # Cerca il primo { ... } che sia JSON valido
    match = re.search(r'\{.*\}', testo, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    raise json.JSONDecodeError("Nessun JSON trovato", testo, 0)


def genera_prompt_musicgen(payload_json: dict):
    attivita = payload_json.get("activity")
    if attivita not in SYSTEM_PROMPTS:
        raise ValueError(f"Attività non riconosciuta: {attivita}")

    # Fallback differenziato per tipo di attività
    FALLBACK = {
        "weightlifting": ("weightlifting_unknown", "heavy electronic, driving beat, high energy, 140 bpm", "ambient drone, muffled bass, drumless"),
        "running":       ("running_unknown", "electronic workout beat, steady rhythm"),
        "yoga_cooldown": ("yoga_unknown", "healing ambient, soft pads, no drums"),
    }

    system_prompt = SYSTEM_PROMPTS[attivita]
    prompt_completo = f"{system_prompt}\n\n Dati in tempo reale:\n{json.dumps(payload_json, indent=2)}"

    try:
        response = model.generate_content(prompt_completo)
        parts = response.candidates[0].content.parts

        # Prova ogni part dall'ultima alla prima finché non trova JSON valido
        risposta_json = None
        for part in reversed(parts):
            try:
                risposta_json = _estrai_json(part.text)
                break
            except (json.JSONDecodeError, AttributeError):
                continue

        if risposta_json is None:
            raise json.JSONDecodeError("Nessuna part conteneva JSON valido", "", 0)

        categoria = risposta_json.get("categoria")

        if attivita == "weightlifting":
            lifting_prompt  = risposta_json.get("lifting_prompt")
            recovery_prompt = risposta_json.get("recovery_prompt")
            if lifting_prompt and recovery_prompt:
                return categoria, lifting_prompt, recovery_prompt
            print("Errore: chiavi lifting/recovery non trovate nel JSON.")
            return FALLBACK["weightlifting"]

        else:
            testo_pulito = risposta_json.get("musicgen_prompt")
            if testo_pulito:
                return categoria, testo_pulito
            print("Errore: chiave 'musicgen_prompt' non trovata.")
            return FALLBACK[attivita]

    except json.JSONDecodeError as e:
        print(f"Errore di parsing JSON: {e}")
        return FALLBACK[attivita]
    except Exception as e:
        print(f"Errore generico: {e}")
        return FALLBACK[attivita]

# TEST
def test():
    payload_test = {
        "activity": "running",
        "user_initial_intent": "Giornata pesante, ma oggi ho leg day. Voglio caricare un sacco e puntare all'ipertrofia, mi serve aggressività pura e distorsione.",
        "current_state": "lifting",
        "last_hr_peak": 140,
        "average_hr": 125,
        "time_in_current_state_sec": 10
    }

    print("Inoltro richiesta a Google AI Studio...")
    if payload_test["activity"] == "weightlifting":
        categoria, prompt_lifting, prompt_recovery = genera_prompt_musicgen(payload_test)
        print(categoria)
        print(prompt_lifting)
        print(prompt_recovery)
    else:
        categoria, risultato = genera_prompt_musicgen(payload_test)
        print(f"\n[OUTPUT PER MUSICGEN]:\n{categoria, risultato}")


if __name__ == "__main__":
    test()