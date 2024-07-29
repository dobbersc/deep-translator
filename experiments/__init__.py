from pathlib import Path
from typing import Final

RESULTS_DIRECTORY: Final[Path] = Path(__file__).parents[1] / "results"

PLOT_EXTENSION: Final[str] = ".pdf"
LONG_FORM: Final[dict[str, str]] = {
    "en": "English",
    "bg": "Bulgarian",
    "cs": "Czech",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "es": "Spanish",
    "et": "Estonian",
    "fi": "Finnish",
    "fr": "French",
    "hu": "Hungarian",
    "it": "Italian",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "nl": "Dutch",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "sk": "Slovak",
    "sl": "Slovene",
    "sv": "Swedish",
}
