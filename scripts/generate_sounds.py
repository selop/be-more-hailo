"""Generate the spoken sound clips with the character's own voice.

    BMO_LANGUAGE=de python scripts/generate_sounds.py            # every German folder
    BMO_LANGUAGE=de python scripts/generate_sounds.py boot ack   # only these

Clips are written unfiltered to sounds/<category>_sounds (English) or
sounds/<category>_sounds_<language>; the agent applies the speaker filter
when it caches them at boot. Run it on the device, where the voice model is.
"""

import os
import sys
import wave

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import LANGUAGE, SOUNDS_DIR  # noqa: E402
from core.tts import PiperSynthesizer, clean_text_for_speech  # noqa: E402

greetings = [
    "Hello! BMO is here!",
    "Who wants to play video games?",
    "BMO is ready for action!",
    "Yay! A new friend!",
    "BMO is online and feeling fine!",
    "Greetings, human!",
    "BMO is powered up!",
    "Let's go on an adventure!",
    "BMO is happy to see you!",
    "System check complete. BMO is perfect!",
    "Hello! What game are we playing today?",
    "BMO is awake!",
    "I am a little living robot!",
    "BMO is ready to help!",
    "Hello! Do you have any batteries?",
    "BMO is here to save the day!",
    "Yay! You're back!",
    "BMO is fully charged!",
    "Hello! Let's do something fun!",
    "BMO is ready to compute!"
]

thinking = [
    "Hmm... BMO is processing...",
    "Let me check my memory banks...",
    "BMO is thinking very hard...",
    "Calculating...",
    "One moment, please...",
    "BMO is searching for the answer...",
    "Let me see...",
    "Processing data...",
    "BMO is computing...",
    "Hold on, BMO is thinking...",
    "Let me look that up...",
    "BMO is accessing the mainframe...",
    "Thinking...",
    "Let me ponder that...",
    "BMO is analyzing the situation...",
    "Just a second...",
    "BMO is running a diagnostic...",
    "Let me figure this out...",
    "BMO is working on it...",
    "Processing..."
]

ack = {"okay": "Okay.", "got_it": "Got it!", "on_it": "On it!"}

boot = [
    "BMO is waking up.",
    "Loading the brain, please wait.",
    "Good morning. BMO is starting up.",
    "Booting BMO systems.",
    "BMO is getting ready.",
    "Hold on, BMO is thinking really hard.",
    "Just a moment, loading BMO brain.",
    "BMO power on sequence initiated.",
    "Warming up the circuits.",
    "BMO is almost ready.",
]

analyzing = [
    "Hmm, that looks rather interesting. Give me a moment.",
    "BMO is developing the film roll, please stand by.",
    "Let BMO take a closer look at this.",
    "Ooh, what do we have here? BMO is analyzing.",
    "Processing pixels. BMO sees something cool.",
    "BMO is squinting really hard at this one.",
    "Hold on, BMO is putting on the reading glasses.",
    "Scanning image data. This is exciting.",
    "BMO is thinking about what they see.",
    "Almost there. BMO is connecting the dots.",
]

TEXTS = {
    "en": {"greeting": greetings, "thinking": thinking, "ack": ack, "boot": boot, "analyzing": analyzing},
    "de": {
        "greeting": [
            "Hallo! BMO ist da!",
            "Wer hat Lust auf Videospiele?",
            "BMO ist bereit für Action!",
            "Juhu! Ein neuer Freund!",
            "BMO ist online und fühlt sich prima!",
            "Sei gegrüßt, Mensch!",
            "BMO ist hochgefahren!",
            "Auf ins Abenteuer!",
            "BMO freut sich, dich zu sehen!",
            "Systemprüfung abgeschlossen. BMO ist perfekt!",
            "Hallo! Welches Spiel spielen wir heute?",
            "BMO ist wach!",
            "Ich bin ein kleiner lebendiger Roboter!",
            "BMO ist bereit zu helfen!",
            "Hallo! Hast du vielleicht Batterien?",
            "BMO ist da und rettet den Tag!",
            "Juhu! Du bist wieder da!",
            "BMO ist voll aufgeladen!",
            "Hallo! Lass uns etwas Lustiges machen!",
            "BMO ist bereit zu rechnen!",
        ],
        "thinking": [
            "Hmm... BMO verarbeitet das...",
            "Ich schaue mal in meinem Speicher nach...",
            "BMO denkt ganz fest nach...",
            "Ich rechne...",
            "Einen Moment, bitte...",
            "BMO sucht nach der Antwort...",
            "Mal sehen...",
            "Daten werden verarbeitet...",
            "BMO rechnet...",
            "Warte kurz, BMO denkt nach...",
            "Das schaue ich mal nach...",
            "BMO greift auf den Großrechner zu...",
            "Ich denke nach...",
            "Darüber muss ich kurz nachdenken...",
            "BMO analysiert die Lage...",
            "Nur eine Sekunde...",
            "BMO führt eine Diagnose durch...",
            "Das kriege ich heraus...",
            "BMO arbeitet daran...",
            "Verarbeitung läuft...",
        ],
        "ack": {"okay": "Okay.", "alles_klar": "Alles klar!", "bin_dran": "Bin dran!"},
        "boot": [
            "BMO wacht auf.",
            "Das Gehirn wird geladen, bitte warten.",
            "Guten Morgen. BMO fährt hoch.",
            "BMO-Systeme werden gestartet.",
            "BMO macht sich bereit.",
            "Warte kurz, BMO denkt ganz fest nach.",
            "Einen Moment, BMOs Gehirn wird geladen.",
            "BMO-Einschaltsequenz gestartet.",
            "Die Schaltkreise werden aufgewärmt.",
            "BMO ist fast bereit.",
        ],
        "analyzing": [
            "Hmm, das sieht ziemlich interessant aus. Gib mir einen Moment.",
            "BMO entwickelt gerade den Film, bitte warten.",
            "Lass BMO das mal genauer ansehen.",
            "Oh, was haben wir denn hier? BMO analysiert.",
            "Pixel werden verarbeitet. BMO sieht etwas Cooles.",
            "BMO kneift die Augen ganz fest zusammen.",
            "Warte kurz, BMO setzt die Lesebrille auf.",
            "Bilddaten werden gescannt. Wie aufregend.",
            "BMO denkt darüber nach, was es da sieht.",
            "Gleich geschafft. BMO verbindet die Punkte.",
        ],
    },
}


def main(categories):
    texts = TEXTS[LANGUAGE]
    suffix = "" if LANGUAGE == "en" else f"_{LANGUAGE}"
    synth = PiperSynthesizer()
    for category in categories or sorted(texts):
        folder = os.path.join(SOUNDS_DIR, f"{category}_sounds{suffix}")
        os.makedirs(folder, exist_ok=True)
        clips = texts[category]
        named = clips.items() if isinstance(clips, dict) else ((f"{category}_{n:02d}", text) for n, text in enumerate(clips, 1))
        for name, text in named:
            path = os.path.join(folder, f"{name}.wav")
            print(f"{path}: {text}")
            # clean_text_for_speech applies the pronunciation fixes (how the voice should say the name)
            pcm = synth.synthesize_raw(clean_text_for_speech(text))
            with wave.open(path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(22050)
                wf.writeframes(pcm)
    print("Done.")


if __name__ == "__main__":
    main(sys.argv[1:])
