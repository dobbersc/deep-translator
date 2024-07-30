
from typing import TYPE_CHECKING

from translator import LOG_SEPARATOR
from translator.datasets import EuroparlCorpus
from translator.models import Translator
from translator.utils.random import set_seed
from translator.utils.torch import detect_device

from experiments import RESULTS_DIRECTORY

if TYPE_CHECKING:
    from pathlib import Path

    import torch


def main() -> None:
    set_seed(42)
    device: torch.device = detect_device()

    source_language: str = "de"
    target_language: str = "en"

    corpus: EuroparlCorpus = EuroparlCorpus.load(source_language, target_language)
    corpus = corpus.downsample(0.1)
    _, _, test_split = corpus.split()

    root_directory: Path = RESULTS_DIRECTORY / "models" / f"{source_language}2{target_language}"
    translators: dict[str, Translator] = {}
    for model_directory in filter(lambda x: x.is_dir(), root_directory.iterdir()):
        translator: Translator = Translator.load(model_directory / "model.pt", map_location=device)
        translator.to(device)
        translators[model_directory.name] = translator

    print(LOG_SEPARATOR)
    for data_point in corpus[:10]:
        print(f"Source: {data_point.source}")
        print(f"Target: {data_point.target}")
        for translator_name, translator in translators.items():
            print(f" - {translator_name}: {translator.translate(data_point.source)}")
        print(LOG_SEPARATOR)


if __name__ == "__main__":
    main()
