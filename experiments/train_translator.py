import functools
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import torch
from gensim.models import KeyedVectors
from sacremoses import MosesDetokenizer

from translator.datasets import EuroparlCorpus, ParallelCorpus
from translator.language import Language
from translator.models import Translator
from translator.tokenizers import Tokenizer, preprocess
from translator.trainer import ModelTrainer
from translator.utils.random import set_seed
from translator.utils.torch import detect_device

from experiments import RESULTS_DIRECTORY

if TYPE_CHECKING:
    from sacrebleu.metrics.bleu import BLEUScore

logger: logging.Logger = logging.getLogger("translator")


def _prepare_languages_and_embeddings(
    embeddings: Literal["end2end", "word2vec"],
    train_split: ParallelCorpus,
    source_tokenizer: Tokenizer,
    target_tokenizer: Tokenizer,
    unk_threshold: int,
) -> tuple[Language, Language, KeyedVectors | None, KeyedVectors | None]:
    source_embeddings: KeyedVectors | None = None
    target_embeddings: KeyedVectors | None = None
    source_language: Language
    target_language: Language

    if embeddings == "end2end":
        source_language = Language.from_sentences(
            name=train_split.source_language,
            sentences=(source_tokenizer(sentence) for sentence in train_split.source_sentences),
            unk_threshold=unk_threshold,
        )
        target_language = Language.from_sentences(
            name=train_split.target_language,
            sentences=(target_tokenizer(sentence) for sentence in train_split.target_sentences),
            unk_threshold=unk_threshold,
        )

    elif embeddings == "word2vec":
        embeddings_directory: Path = RESULTS_DIRECTORY / "embeddings"
        source_embeddings = KeyedVectors.load_word2vec_format(
            str(embeddings_directory / train_split.source_language / "vectors.txt"),
        )
        target_embeddings = KeyedVectors.load_word2vec_format(
            str(embeddings_directory / train_split.target_language / "vectors.txt"),
        )
        source_language = Language.from_embeddings(
            name=train_split.source_language,
            embeddings=source_embeddings,
        )
        target_language = Language.from_embeddings(
            name=train_split.target_language,
            embeddings=target_embeddings,
        )

    else:
        raise AssertionError

    return source_language, target_language, source_embeddings, target_embeddings


def train(
    source_language_name: str,
    target_language_name: str,
    out_directory: Path,
    *,
    downsample_corpus: float | None = 0.1,
    segmentation_level: Literal["word", "character"] = "word",
    lowercase: bool = True,
    embeddings: Literal["end2end", "word2vec"] = "end2end",
    unk_threshold: int = 5,
    max_epochs: int = 10,
    learning_rate: float = 0.001,
    batch_size: int = 128,
    attention: bool = False,
    num_workers: int = 4,
    seed: int = 42,
    device: str | torch.device | None = None,
) -> None:
    set_seed(seed)

    if device is None:
        device = detect_device()

    # Read training data and split into train, dev and test splits.
    corpus: EuroparlCorpus = EuroparlCorpus.load(source_language_name, target_language_name)
    if downsample_corpus is not None:
        corpus = corpus.downsample(downsample_corpus)
    train_split, dev_split, test_split = corpus.split()

    # Define tokenizers for the source and target language.
    source_tokenizer: Tokenizer = functools.partial(
        preprocess,
        language=corpus.source_language,
        segmentation_level=segmentation_level,
        lowercase=lowercase,
    )
    target_tokenizer: Tokenizer = functools.partial(
        preprocess,
        language=corpus.target_language,
        segmentation_level=segmentation_level,
        lowercase=lowercase,
    )

    # Make vocabulary dictionaries for both languages and pretrained embeddings (if specified).
    (
        source_language,
        target_language,
        source_pretrained_embeddings,
        target_pretrained_embeddings,
    ) = _prepare_languages_and_embeddings(
        embeddings=embeddings,
        train_split=train_split,
        source_tokenizer=source_tokenizer,
        target_tokenizer=target_tokenizer,
        unk_threshold=unk_threshold,
    )

    # Initialize the translator model.
    translator_kwargs: dict[str, Any] = (
        {"attention": True} if attention else {"attention": False, "bidirectional_encoder": True}
    )
    if segmentation_level == "word":
        translator_kwargs["target_detokenizer"] = MosesDetokenizer(target_language.name).detokenize
    elif segmentation_level == "character":
        translator_kwargs["target_detokenizer"] = "".join

    translator: Translator = Translator(
        source_language=source_language,
        target_language=target_language,
        source_tokenizer=source_tokenizer,
        target_tokenizer=target_tokenizer,
        source_embedding_size=300,  # Same for pretrained embeddings.
        target_embedding_size=300,  # Same for pretrained embeddings.
        source_pretrained_embeddings=source_pretrained_embeddings,
        target_pretrained_embeddings=target_pretrained_embeddings,
        embedding_dropout=0.1,
        **translator_kwargs,
    )
    translator.to(device)

    # Train the model.
    model_trainer: ModelTrainer = ModelTrainer(translator, train_split, dev_split, test_split)
    model_trainer.train(
        max_epochs=max_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # Evaluate and save the model.
    bleu: BLEUScore = translator.evaluate_bleu(
        sources=test_split.source_sentences,
        targets=[[t] for t in test_split.target_sentences],
        lowercase=lowercase,
        batch_size=batch_size,
    )
    logger.info("TEST  %r", bleu)

    out_directory.mkdir(parents=True, exist_ok=True)
    translator.save(out_directory / "model.pt")


def main(source_language: str = "de", target_language: str = "en") -> None:
    # Sometimes needed to avoid "RuntimeError: received 0 items of ancdata"
    torch.multiprocessing.set_sharing_strategy("file_system")

    models_directory: Path = RESULTS_DIRECTORY / "models" / f"{source_language}2{target_language}"

    configurations: list[tuple[Path, dict[str, Any]]] = []
    for configuration_string in (
        "character_end2end_embeddings_with_attention",
        "character_end2end_embeddings_without_attention",
        "word_end2end_embeddings_with_attention",
        "word_end2end_embeddings_without_attention",
        "word_word2vec_embeddings_with_attention",
        "word_word2vec_embeddings_without_attention",
    ):
        segmentation_level, embeddings, _, attention_string, _ = configuration_string.split("_")
        attention: bool = attention_string == "with"
        configurations.append(
            (
                models_directory / configuration_string,
                {"segmentation_level": segmentation_level, "embeddings": embeddings, "attention": attention},
            ),
        )

    for out_directory, configuration in configurations:
        train(
            source_language_name=source_language,
            target_language_name=target_language,
            out_directory=out_directory,
            **configuration,
        )


if __name__ == "__main__":
    main()
