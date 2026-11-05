"""Shared settings for the Lugbara extension of the Rutooro multilingual translator."""
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE_MODEL = ROOT / "source-model"          # copy of mmuniversity/rutooro-multilingual-translator
OUTPUT_MODEL = ROOT / "lugbara-model"         # fine-tuned model that gets uploaded
DATA = ROOT / "data"
ANALYSIS = ROOT / "analysis"

SOURCE_REPO = "mmuniversity/rutooro-multilingual-translator"
TARGET_REPO = "muni-university/lugbara-multilingual-translator"
SALT_REPO = "Sunbird/salt"

# language tag used in the prompt -> SALT column
LANGS = {
    "lugbara": "lgg_text",
    "rutooro": "ttj_text",
    "luganda": "lug_text",
    "acholi": "ach_text",
    "runyankore": "nyn_text",
}
NEW_LANG = "lugbara"

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
