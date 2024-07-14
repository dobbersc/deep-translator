import functools
import random
import string
from collections import Counter
from collections.abc import Sequence
from typing import Any, Literal, cast

import matplotlib.pyplot as plt
import numpy as np
import spacy
from spacy.language import Language
from tqdm import tqdm

from translator.datasets import DataPoint, EuroparlCorpus
from translator.preprocessing import preprocess

from experiments import LONG_FORM, PREFERRED_PLOT_EXTENSION, RESULTS_PATH

PLOTS_PATH = RESULTS_PATH / "plots"


def count_(
    mode: Literal["word", "word_length", "sentence_length", "pos_tags"],
    sentence: str,
    language: str,
    tagger: Language | None = None,
) -> Counter[int] | Counter[str]:
    """Based on the mode counts different aspcets of the input setence.

    Args:
        mode: Based on the mode a different counter is returned.
        sentence: The sentence based on which the counter will be created.
        language: The language in which the input sentence is written.
        tagger: Spacy pipeline to extract POS-tag.

    Returns:
        Counter containing different counts based on mode.
    """
    if mode == "word_length":
        tokens = preprocess(sentence, language=language)
        return Counter([len(token) for token in tokens])
    if mode == "word":
        tokens = preprocess(sentence, language=language)
        return Counter(tokens)
    if mode == "sentence_length":
        tokens = preprocess(sentence, language=language)
        return Counter([len(tokens)])
    if mode == "pos_tags":
        if tagger is not None:
            doc = tagger(sentence)
            return Counter([token.tag_ for token in doc])
        msg = "The tagger can't be set to None when the mode is 'pos_tags'"
        raise ValueError(msg)
    return Counter()  # type: ignore[unreachable]


def apply_threshold(counter: Counter[Any], threshold: int) -> Counter[Any]:
    """Removes the elements of the counter whose count is smaller than the threshold.

    Args:
        counter: The counter that will be filtered.
        threshold: The threshold used for filtering the counter.

    Returns:
        Counter with entries below the threshold removed.
    """
    return Counter({key: count for key, count in counter.items() if count >= threshold})


def remove_punctuation(counter: Counter[Any]) -> Counter[Any]:
    """Removes the elements that count punctuations.

    Args:
        counter: Counter that will be filtered.

    Returns:
        Counter with entries counting punctuations removed.
    """
    return Counter({key: count for key, count in counter.items() if key not in string.punctuation})


@functools.lru_cache(maxsize=1)
def load_english_spacy_tagger() -> Language:
    """Loads the english spacy pipeline and excludes some not needed components.

    Returns:
        English spacy pipeline for POS-tagging.
    """
    return spacy.load(
        "en_core_web_sm",
        exclude=("parser", "senter", "attribute_ruler", "lemmatizer", "ner"),
    )


@functools.lru_cache(maxsize=1)
def load_german_spacy_tagger() -> Language:
    """Loads the german spacy pipeline and excludes some not needed components.

    Returns:
        German spacy pipeline for POS-tagging.
    """
    return spacy.load(
        "de_core_news_sm",
        exclude=("morphologizer", "parser", "lemmatizer", "senter", "attribute_ruler", "ner"),
    )


def load_spacy_tagger(language: str) -> Language:
    """Loads the spacy pipeline corresponding to the language provided.

    Args:
        language: Determines the spacy pipeline that is loaded.

    Raises:
        ValueError: In case the specified language is not supported.

    Returns:
        Spacy pipeline for POS-tagging.
    """
    if language == "en":
        return load_english_spacy_tagger()
    if language == "de":
        return load_german_spacy_tagger()
    msg = f"Language {language} is not supported."
    raise ValueError(msg)


def part_of_speech_tags(
    corpus: EuroparlCorpus | DataPoint | Sequence[DataPoint],
    source_language: str,
    target_language: str,
    k: int = 10,
) -> None:
    """Plots the k most common POS-tag for each language.

    Args:
        corpus: EuroparlCorpus, a EuroparlCorpus slice or a EuroparlCorpus data point.
        source_language: Souce language.
        target_language: Target language.
        k: Number of POS-tags to plot. Defaults to 10.
    """
    source_tagger = load_spacy_tagger(source_language)
    target_tagger = load_spacy_tagger(target_language)
    counters: dict[str, Counter[int]] = {source_language: Counter(), target_language: Counter()}
    for sentence_pair in tqdm(corpus):
        data_point = cast(DataPoint, sentence_pair)
        counters[source_language].update(
            count_("pos_tags", data_point.source_sentence, source_language, tagger=source_tagger),  # type: ignore[arg-type]
        )
        counters[target_language].update(
            count_("pos_tags", data_point.target_sentence, target_language, tagger=target_tagger),  # type: ignore[arg-type]
        )

    fig, _ = plt.subplots(nrows=2, ncols=1, sharey=True)

    for i, language in enumerate([source_language, target_language]):
        items = counters[language].most_common(k)
        labels, values = zip(*items, strict=False)
        ax = fig.get_axes()[i]
        x = np.arange(1, len(labels) + 1)
        ax.bar(x, values)
        ax.set_xticks(x, labels, rotation=30, ha="right")
        ax.set_ylabel("Frequency")
        ax.set_title(f"{LONG_FORM[language]}")

    from_to = LONG_FORM[source_language] + "-" + LONG_FORM[target_language]
    fig.suptitle(f"Top-{k} Most Frequent POS-Tags in {from_to} Parallel Corpus")
    fig.tight_layout()
    filename = f"top_{k}_pos_tags_{source_language}_{target_language}{PREFERRED_PLOT_EXTENSION}"
    plt.savefig(PLOTS_PATH / filename)
    plt.close(fig)


def length_(
    corpus: EuroparlCorpus | DataPoint | Sequence[DataPoint],
    source_language: str,
    target_language: str,
    mode: Literal["word", "sentence"] = "word",
    threshold: int = 1,
) -> None:
    """Depending on the mode, plots either the word or sentence length distibution as a bar plot.

    Args:
        corpus: EuroparlCorpus, a EuroparlCorpus slice or a EuroparlCorpus data point.
        source_language: Souce language.
        target_language: Target language.
        mode: Based on the mode calculates either word or sentence length distribution. Defaults to "word".
        threshold: Counts below the threshold are discarded. Defaults to 1.
    """
    counters: dict[str, Counter[int]] = {source_language: Counter(), target_language: Counter()}
    for sentence_pair in tqdm(corpus):
        data_point = cast(DataPoint, sentence_pair)
        counters[source_language].update(count_(f"{mode}_length", data_point.source_sentence, source_language))  # type: ignore[arg-type]
        counters[target_language].update(count_(f"{mode}_length", data_point.target_sentence, target_language))  # type: ignore[arg-type]

    fig, _ = plt.subplots(nrows=2, ncols=1, sharey=True, sharex=True)
    max_value = -1
    print(counters)
    for i, language in enumerate([source_language, target_language]):
        items = apply_threshold(counters[language], threshold=threshold).items()
        labels, values = zip(*items, strict=False)
        ax = fig.get_axes()[i]
        ax.bar(labels, values)
        ax.set_ylabel("Frequency")
        ax.set_title(f"{LONG_FORM[language]}")
        max_value = max(max_value, *labels)

    step = 1 if mode == "word" else 10
    plt.xticks(np.arange(0, max_value + 1, step))
    plt.xlabel(f"{mode.capitalize()} Length")

    fig.suptitle(
        (
            f"{mode.capitalize()} length distribution "
            f"for {LONG_FORM[source_language]}-{LONG_FORM[target_language]} "
            "Parallel Corpus"
        ),
    )

    fig.tight_layout()
    filename = f"{mode}_length_{threshold}_{source_language}_{target_language}{PREFERRED_PLOT_EXTENSION}"
    plt.savefig(PLOTS_PATH / filename)
    plt.close(fig)


def most_frequent_words(
    corpus: EuroparlCorpus | DataPoint | Sequence[DataPoint],
    source_language: str,
    target_language: str,
    k: int = 20,
) -> None:
    """Plots the k most frequent words for each language.

    Args:
        corpus: corpus: EuroparlCorpus, a EuroparlCorpus slice or a EuroparlCorpus data point.
        source_language (str): Source language.
        target_language (str): Target language.
        k: The k most frequent words are plotted. Defaults to 20.
    """
    counters: dict[str, Counter[str]] = {source_language: Counter(), target_language: Counter()}
    for sentence_pair in tqdm(corpus):
        data_point = cast(DataPoint, sentence_pair)
        counters[source_language].update(count_("word", data_point.source_sentence, source_language))  # type: ignore[arg-type]
        counters[target_language].update(count_("word", data_point.target_sentence, target_language))  # type: ignore[arg-type]

    fig, _ = plt.subplots(nrows=2, ncols=1, sharey=True)

    for i, language in enumerate([source_language, target_language]):
        items = remove_punctuation(counters[language]).most_common(k)
        labels, values = zip(*items, strict=False)
        ax = fig.get_axes()[i]
        x = np.arange(1, len(labels) + 1)
        ax.bar(x, values)
        ax.set_xticks(x, labels, rotation=30, ha="right")
        ax.set_ylabel("Frequency")
        ax.set_title(f"{LONG_FORM[language]}")

    fig.suptitle(
        (
            f"Top-{k} Most Frequent Words "
            f"in {LONG_FORM[source_language]}-{LONG_FORM[target_language]} "
            "Parallel Corpus"
        ),
    )
    fig.tight_layout()
    filename = f"top_{k}_most_frequent_{source_language}_{target_language}{PREFERRED_PLOT_EXTENSION}"
    plt.savefig(PLOTS_PATH / filename)
    plt.close(fig)


def main() -> None:
    random.seed(42)
    corpus: EuroparlCorpus = EuroparlCorpus.load(source_language="en", target_language="de")

    most_frequent_words(corpus, corpus.source_language, corpus.target_language, k=15)
    length_(corpus, corpus.source_language, corpus.target_language, mode="word", threshold=100000)
    length_(corpus, corpus.source_language, corpus.target_language, mode="sentence", threshold=310)
    part_of_speech_tags(random.sample(corpus, 100000), corpus.source_language, corpus.target_language, k=10)


if __name__ == "__main__":
    PLOTS_PATH.mkdir(parents=True, exist_ok=True)
    main()
