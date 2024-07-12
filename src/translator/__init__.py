import os
from pathlib import Path
from typing import Final

__version__: Final[str] = "0.0.1"

CACHE_DIRECTORY: Path = Path(os.getenv("DEEP_TRANSLATOR_CACHE") or Path.home() / ".cache" / "deep_translator")
