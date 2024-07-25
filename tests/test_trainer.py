import functools
from pathlib import Path

import gensim.downloader
import pytest
from gensim.models import KeyedVectors

from translator.datasets import ParallelCorpus
from translator.language import Language
from translator.models import Translator
from translator.tokenizers import Tokenizer, preprocess
from translator.trainer import ModelTrainer
from translator.utils.download import download_from_url
from translator.utils.random import set_seed
from translator.utils.torch import detect_device


@pytest.fixture()
def europarl_small(resources_dir: Path) -> ParallelCorpus:
    source_sentences: list[str] = (resources_dir / "europarl_small.de").read_text().splitlines()
    target_sentences: list[str] = (resources_dir / "europarl_small.en").read_text().splitlines()
    return ParallelCorpus(source_sentences, target_sentences, source_language="de", target_language="en")


class TestModelTrainer:
    @pytest.mark.integration()
    def test_train_translator(self, europarl_small: ParallelCorpus) -> None:
        set_seed(42)

        train_split, dev_split, test_split = europarl_small, europarl_small, europarl_small

        source_tokenizer: Tokenizer = functools.partial(
            preprocess,
            language=europarl_small.source_language,
            lowercase=True,
        )
        target_tokenizer: Tokenizer = functools.partial(
            preprocess,
            language=europarl_small.target_language,
            lowercase=True,
        )

        source_language: Language = Language.from_sentences(
            name=europarl_small.source_language,
            sentences=(source_tokenizer(sentence) for sentence in train_split.source_sentences),
        )
        target_language: Language = Language.from_sentences(
            name=europarl_small.target_language,
            sentences=(target_tokenizer(sentence) for sentence in train_split.target_sentences),
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

    @pytest.mark.integration()
    def test_train_translator_with_pretrained_embeddings(self, europarl_small: ParallelCorpus, tmp_path: Path) -> None:
        set_seed(42)

        train_split, dev_split, test_split = europarl_small, europarl_small, europarl_small

        german_word2vec: Path = tmp_path / "embeddings" / "german.model"
        german_word2vec.parent.mkdir(parents=True, exist_ok=True)
        download_from_url("https://cloud.devmount.de/d2bc5672c523b086/german.model", german_word2vec)

        source_pretrained_embeddings: KeyedVectors = gensim.downloader.load("word2vec-google-news-300")
        target_pretrained_embeddings: KeyedVectors = KeyedVectors.load_word2vec_format(
            str(german_word2vec),
            binary=True,
        )

        source_tokenizer: Tokenizer = functools.partial(
            preprocess,
            language=europarl_small.source_language,
            lowercase=False,
        )
        target_tokenizer: Tokenizer = functools.partial(
            preprocess,
            language=europarl_small.target_language,
            lowercase=False,
        )

        source_language: Language = Language.from_embeddings(
            name=europarl_small.source_language,
            embeddings=source_pretrained_embeddings,
        )
        target_language: Language = Language.from_embeddings(
            name=europarl_small.target_language,
            embeddings=target_pretrained_embeddings,
        )

        translator: Translator = Translator(
            source_language=source_language,
            target_language=target_language,
            source_tokenizer=source_tokenizer,
            target_tokenizer=target_tokenizer,
            source_embedding_size=300,
            target_embedding_size=300,
            source_pretrained_embeddings=source_pretrained_embeddings,
            target_pretrained_embeddings=target_pretrained_embeddings,
        )
        translator.to(detect_device())

        model_trainer: ModelTrainer = ModelTrainer(translator, train_split, dev_split, test_split)
        test_perplexity: float = model_trainer.train(max_epochs=20, learning_rate=0.005, batch_size=1)
        assert 1.0 <= test_perplexity <= 20.0
