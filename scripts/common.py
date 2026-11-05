"""Shared settings for the Lugbara/Luganda extension of the Rutooro multilingual translator."""
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE_MODEL = ROOT / "source-model"          # copy of mmuniversity/rutooro-multilingual-translator
OUTPUT_MODEL = ROOT / "lugbara-model"         # fine-tuned model that gets uploaded
DATA = ROOT / "data"
ANALYSIS = ROOT / "analysis"

SOURCE_REPO = "mmuniversity/rutooro-multilingual-translator"
TARGET_REPO = "ckuma/lugbara-multilingual-translator"   # org "Muni University" (web name ckuma)
SALT_REPO = "Sunbird/salt"

# language name -> SALT column
COLS = {
    "english": "eng_source_text",
    "lugbara": "lgg_text",
    "luganda": "lug_text",
    "rutooro": "ttj_text",
    "acholi": "ach_text",
    "runyankore": "nyn_text",
}

# translation directions the model is trained on: (source, target, training pairs; None = all)
# the prompt tag always names the TARGET language, e.g. ">>english<< <Lugbara text>"
DIRECTIONS = [
    ("english", "lugbara", None),
    ("lugbara", "english", None),
    ("english", "luganda", 6000),
    ("luganda", "english", 16000),
    ("english", "rutooro", 2000),      # replay: keep what the Rutooro model already does
    ("english", "acholi", 2000),
    ("english", "runyankore", 2000),
]
TAGS = sorted({t for _, t, _ in DIRECTIONS})


def dname(src, tgt):
    return f"{src}-{tgt}"


# Lugbara writes implosives / glottalised consonants with an apostrophe ('b, 'd, 'y).
# Sources mix ' ’ ‘ ʼ ` ´ ; fold them all to the ASCII apostrophe the tokenizer knows.
APOSTROPHES = "’‘ʼʻ`´′"
_APOS_RE = re.compile(f"[{APOSTROPHES}]")
_WS_RE = re.compile(r"\s+")


def normalize(text: str) -> str:
    if text is None:
        return ""
    text = unicodedata.normalize("NFC", str(text))
    text = _APOS_RE.sub("'", text)
    text = text.replace("“", '"').replace("”", '"')
    return _WS_RE.sub(" ", text).strip()


def tag(lang: str) -> str:
    return f">>{lang}<<"
