import functools
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from gensim.models import KeyedVectors, Word2Vec
from sacremoses import MosesDetokenizer

from translator.datasets import ParallelCorpus
from translator.language import Language
from translator.models import Translator
from translator.tokenizers import Tokenizer, preprocess
from translator.trainer import ModelTrainer
from translator.utils.random import set_seed
from translator.utils.torch import detect_device

if TYPE_CHECKING:
    from collections.abc import Sequence

    from sacrebleu.metrics.bleu import BLEUScore


@pytest.fixture()
def europarl_small(resources_dir: Path) -> ParallelCorpus:
    source_sentences: list[str] = (resources_dir / "europarl_small.de").read_text().splitlines()
    target_sentences: list[str] = (resources_dir / "europarl_small.en").read_text().splitlines()
    return ParallelCorpus(source_sentences, target_sentences, source_language="de", target_language="en")


def augment_train_data(corpus: ParallelCorpus, times: int) -> ParallelCorpus:
    return ParallelCorpus(
        source_sentences=[s for s in corpus.source_sentences for _ in range(times)],
        target_sentences=[t for t in corpus.target_sentences for _ in range(times)],
        source_language=corpus.source_language,
        target_language=corpus.target_language,
    )


class TestModelTrainer:
    @pytest.mark.integration()
    @pytest.mark.parametrize(
        "translator_kwargs",
        [
            {},
            {"attention": True},
            {"bidirectional_encoder": True},
            {"encoder_num_layers": 4, "decoder_num_layers": 2, "hidden_size": 64},
            {
                "attention": True,
                "propagate_hidden_and_cell_state": False,
                "bidirectional_encoder": True,
                "encoder_num_layers": 1,
                "decoder_num_layers": 2,
                "hidden_size": 64,
            },
        ],
        ids=("default", "with_attention", "bidirectional_encoder", "multi_layer", "all"),
    )
    def test_train_translator(self, europarl_small: ParallelCorpus, translator_kwargs: dict[str, Any]) -> None:
        set_seed(42)

        # Read training data and split into train, dev and test splits.
        train_split, dev_split, test_split = europarl_small, europarl_small, europarl_small
        train_split = augment_train_data(train_split, times=10)

        # Define tokenizers for the source and target language.
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

        # Make vocabulary dictionaries for both languages.
        source_language: Language = Language.from_sentences(
            name=europarl_small.source_language,
            sentences=(source_tokenizer(sentence) for sentence in train_split.source_sentences),
        )
        target_language: Language = Language.from_sentences(
            name=europarl_small.target_language,
            sentences=(target_tokenizer(sentence) for sentence in train_split.target_sentences),
        )

        # Initialize the translator model.
        translator: Translator = Translator(
            source_language=source_language,
            target_language=target_language,
            source_tokenizer=source_tokenizer,
            target_tokenizer=target_tokenizer,
            target_detokenizer=MosesDetokenizer(target_language.name).detokenize,
            **translator_kwargs,
        )
        translator.to(detect_device())

        # Train the model.
        model_trainer: ModelTrainer = ModelTrainer(translator, train_split, dev_split, test_split)
        test_perplexity: float = model_trainer.train(max_epochs=10, learning_rate=0.01, batch_size=2)
        assert 1.0 <= test_perplexity <= 1.05

        bleu: BLEUScore = translator.evaluate_bleu(
            sources=test_split.source_sentences,
            targets=[[t] for t in test_split.target_sentences],
            lowercase=True,
        )
        assert bleu.score >= 99

        # Save and reload the model.
        with BytesIO() as buffer:
            translator.save(buffer)
            buffer.seek(0)
            translator = Translator.load(buffer)

        # Translate an example sentence.
        translation: str = translator.translate("Frau Präsidentin, zur Geschäftsordnung.")
        expected: str = "Madam President, on a point of order.".lower()
        assert translation == expected

    @pytest.mark.integration()
    def test_train_translator_with_pretrained_embeddings(self, europarl_small: ParallelCorpus) -> None:
        set_seed(42)

        # Read training data and split into train, dev and test splits.
        train_split, dev_split, test_split = europarl_small, europarl_small, europarl_small
        train_split = augment_train_data(train_split, times=10)

        # Define tokenizers for the source and target language.
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

        # Train small Word2Vec embeddings.
        embedding_kwargs: dict[str, Any] = {"vector_size": 256, "window": 1, "min_count": 1, "workers": 1}
        tokenized_sources: list[Sequence[str]] = [source_tokenizer(s) for s in train_split.source_sentences]
        tokenized_targets: list[Sequence[str]] = [target_tokenizer(t) for t in train_split.target_sentences]
        source_pretrained_embeddings: KeyedVectors = Word2Vec(tokenized_sources, **embedding_kwargs).wv
        target_pretrained_embeddings: KeyedVectors = Word2Vec(tokenized_targets, **embedding_kwargs).wv

        # Make vocabulary dictionaries for both languages.
        source_language: Language = Language.from_embeddings(
            name=europarl_small.source_language,
            embeddings=source_pretrained_embeddings,
        )
        target_language: Language = Language.from_embeddings(
            name=europarl_small.target_language,
            embeddings=target_pretrained_embeddings,
        )

        # Initialize the translator model.
        translator: Translator = Translator(
            source_language=source_language,
            target_language=target_language,
            source_tokenizer=source_tokenizer,
            target_tokenizer=target_tokenizer,
            target_detokenizer=MosesDetokenizer(target_language.name).detokenize,
            source_embedding_size=source_pretrained_embeddings.vector_size,
            target_embedding_size=target_pretrained_embeddings.vector_size,
            source_pretrained_embeddings=source_pretrained_embeddings,
            target_pretrained_embeddings=target_pretrained_embeddings,
            freeze_pretrained_embeddings=False,
        )
        translator.to(detect_device())

        # Train the model.
        model_trainer: ModelTrainer = ModelTrainer(translator, train_split, dev_split, test_split)
        test_perplexity: float = model_trainer.train(max_epochs=10, learning_rate=0.01, batch_size=2)
        assert 1.0 <= test_perplexity <= 1.05

        bleu: BLEUScore = translator.evaluate_bleu(
            sources=test_split.source_sentences,
            targets=[[t] for t in test_split.target_sentences],
            lowercase=True,
        )
        assert bleu.score >= 99

        # Save and reload the model.
        with BytesIO() as buffer:
            translator.save(buffer)
            buffer.seek(0)
            translator = Translator.load(buffer)

        # Translate an example sentence.
        translation: str = translator.translate("Frau Präsidentin, zur Geschäftsordnung.")
        expected: str = "Madam President, on a point of order.".lower()
        assert translation == expected
