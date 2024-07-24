import functools
from pathlib import Path

import pytest

from translator.datasets import ParallelCorpus
from translator.language import Language
from translator.models import Translator
from translator.preprocessing import Tokenizer, preprocess
from translator.trainer import ModelTrainer
from translator.utils.random import set_seed
from translator.utils.torch import detect_device


@pytest.fixture()
def sample_corpus(resources_dir: Path) -> ParallelCorpus:
    source_sentences: list[str] = (resources_dir / "europarl_small.de").read_text().splitlines()
    target_sentences: list[str] = (resources_dir / "europarl_small.en").read_text().splitlines()
    return ParallelCorpus(source_sentences, target_sentences, source_language="de", target_language="en")


@pytest.mark.integration()
def test_model_trainer(sample_corpus: ParallelCorpus) -> None:
    set_seed(42)

    train_split, dev_split, test_split = sample_corpus, sample_corpus, sample_corpus

    source_tokenizer: Tokenizer = functools.partial(preprocess, language=sample_corpus.source_language, lowercase=True)
    target_tokenizer: Tokenizer = functools.partial(preprocess, language=sample_corpus.target_language, lowercase=True)

    source_language: Language = Language.from_sentences(
        sample_corpus.source_language,
        (source_tokenizer(sentence) for sentence in train_split.source_sentences),
    )
    target_language: Language = Language.from_sentences(
        sample_corpus.target_language,
        (target_tokenizer(sentence) for sentence in train_split.target_sentences),
    )

    translator: Translator = Translator(
        source_language=source_language,
        target_language=target_language,
        source_tokenizer=source_tokenizer,
        target_tokenizer=target_tokenizer,
    )
    translator.to(detect_device())

    model_trainer: ModelTrainer = ModelTrainer(translator, train_split, dev_split, test_split)
    test_perplexity: float = model_trainer.train(max_epochs=20, learning_rate=0.01, batch_size=2)
    assert 1.0 <= test_perplexity <= 1.05
