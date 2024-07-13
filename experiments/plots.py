import string
from collections import Counter
from collections.abc import Sequence
from typing import Any, Literal, cast

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from translator.datasets import DataPoint, EuroparlCorpus
from translator.preprocessing import preprocess

from experiments import LONG_FORM, PREFERRED_PLOT_EXTENSION, RESULTS_PATH

PLOTS_PATH = RESULTS_PATH / "plots"


def count_(
    mode: Literal["word", "word_length", "sentence_length"],
    sentence: str,
    language: str,
) -> Counter[int] | Counter[str]:
    """Based on the mode counts different aspcets of the input setence.

    Args:
        mode: Based on the mode a different counter is returned.
        sentence: The sentence based on which the counter will be created.
        language: The language in which the input sentence is written.

    Returns:
        Counter containing different counts based on mode.
    """
    tokens = preprocess(sentence, language=language)
    if mode == "word_length":
        return Counter([len(token) for token in tokens])
    if mode == "word":
        return Counter(tokens)
    if mode == "sentence_length":
        return Counter([len(preprocess(text=sentence, language=language))])
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
    corpus: EuroparlCorpus = EuroparlCorpus.load(source_language="en", target_language="de")

    most_frequent_words(corpus, corpus.source_language, corpus.target_language, k=15)
    length_(corpus, corpus.source_language, corpus.target_language, mode="word", threshold=100000)
    length_(corpus, corpus.source_language, corpus.target_language, mode="sentence", threshold=310)


if __name__ == "__main__":
    PLOTS_PATH.mkdir(parents=True, exist_ok=True)
    main()
