from translator import LOG_SEPARATOR
from translator.datasets import EuroparlCorpus
from translator.models import PivotTranslator, Translator
from translator.utils.random import set_seed

from experiments import RESULTS_DIRECTORY


def main() -> None:
    set_seed(42)

    source_translator: Translator = Translator.load(
        RESULTS_DIRECTORY / "models" / "de2en" / "word_word2vec_embeddings_without_attention" / "model.pt",
    )
    target_translator: Translator = Translator.load(
        RESULTS_DIRECTORY / "models" / "en2el" / "word_end2end_embeddings_with_attention" / "model.pt",
    )

    translator: PivotTranslator = PivotTranslator(source_translator, target_translator)

    # Since we don't have a parallel corpus from German to Greek,
    # we translate the German sources and inspect the result by hand.
    corpus: EuroparlCorpus = EuroparlCorpus.load(translator.source_language, target_language="en")
    corpus = corpus.downsample(0.1)
    _, _, test_split = corpus.split()

    print(LOG_SEPARATOR)
    for source in corpus.source_sentences[:10]:
        print(f"Source:      {source}")
        print(f"Translation: {translator.translate(source)}")
        print(LOG_SEPARATOR)


if __name__ == "__main__":
    main()
