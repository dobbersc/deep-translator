import json
import random
import string
import warnings
from collections import Counter
from collections.abc import Sequence
from statistics import fmean
from typing import Any, Literal, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from spacy.language import Language
from tqdm import tqdm

from translator.datasets import DataPoint, EuroparlCorpus
from translator.tokenizers import preprocess

from experiments import LONG_FORM, PLOT_EXTENSION, RESULTS_DIRECTORY
from experiments.spacy_taggers import load_spacy_tagger

PLOTS_PATH = RESULTS_DIRECTORY / "plots"
RAW_COUNTERS_PATH = RESULTS_DIRECTORY / "raw_counters"
MODES = Literal["word", "word_length", "sentence_length", "pos_tags", "sentence_lenth_difference"]


def count_(
    mode: MODES,
    sentence: str,
    language: str,
    tagger: Language | None = None,
    second_sentence: str | None = None,
    second_language: str | None = None,
) -> Counter[int] | Counter[str]:
    """Based on the mode counts different aspcets of the input sentence.

    Args:
        mode: Based on the mode a different counter is returned.
        sentence: The sentence based on which the counter will be created.
        language: The language in which the input sentence is written.
        tagger: Spacy pipeline to extract POS-tag. Defaults to None.
        second_sentence: The second sentence in case the mode requires one. Defaults to None.
        second_language: The language in which the second sentence is written. Defaults to None.

    Raises:
        ValueError: In case the input parameters combination is wrong.

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
            return Counter([token.pos_ for token in doc])
        msg = f"The tagger can't be set to None when the mode is '{mode}'"
        raise ValueError(msg)
    if mode == "sentence_lenth_difference":
        if second_sentence is not None and second_language is not None:
            tokens_first_sentence = preprocess(sentence, language=language)
            tokens_second_sentence = preprocess(second_sentence, language=language)
            return Counter([len(tokens_first_sentence) - len(tokens_second_sentence)])
        msg = f"The second sentence or langugage can't be set to None when the mode is '{mode}'"
        raise ValueError(msg)
    return Counter()  # type: ignore[unreachable]


def _fill_counters(
    counters: dict[str, Counter[Any]] | Counter[Any],
    corpus: EuroparlCorpus | DataPoint | Sequence[DataPoint],
    source_language: str,
    target_language: str,
    *,
    mode: MODES,
    tagger_source: Language | None = None,
    tagger_target: Language | None = None,
) -> dict[str, Counter[Any]] | Counter[Any]:
    """Based on the mode fills the counters that are passed with the corresponding values.

    Args:
        counters: Either a single Counter or a dictionary of Counters (one for each language).
        corpus: EuroparlCorpus, a EuroparlCorpus slice or a EuroparlCorpus data point.
        source_language: Souce language.
        target_language: Target language.
        mode: Determines the values with which the counters are filled.
        tagger_source: Only relevant for POS-tagging. Defaults to None.
        tagger_target: Only relevant for POS-tagging. Defaults to None.

    Returns:
        The filled Counter(s).
    """
    if isinstance(counters, Counter):
        for sentence_pair in tqdm(corpus):
            data_point = cast(DataPoint, sentence_pair)
            counters.update(
                count_(
                    mode=mode,
                    sentence=data_point.source,
                    language=source_language,
                    second_sentence=data_point.target,
                    second_language=target_language,
                ),
            )
    else:
        for sentence_pair in tqdm(corpus):
            data_point = cast(DataPoint, sentence_pair)
            counters[source_language].update(
                count_(
                    mode=mode,
                    sentence=data_point.source,
                    language=source_language,
                    tagger=tagger_source,
                ),
            )
            counters[target_language].update(
                count_(
                    mode=mode,
                    sentence=data_point.target,
                    language=target_language,
                    tagger=tagger_target,
                ),
            )

    return counters


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


def _barplot_categorical(labels: tuple[str], values: tuple[int], ax: Axes, *, title: str = "") -> None:
    """Plots barplot for categorical values.

    Args:
        labels: Categorical values.
        values: Values corresponding to each category.
        ax: On this axis the plot is created.
        title: The axis title, which is optional. Defaults to None.
    """
    x = np.arange(1, len(labels) + 1)
    ax.bar(x, values)
    ax.set_xticks(x, labels, rotation=30, ha="right")
    ax.set_ylabel("Frequency")
    ax.set_title(title)


def _barplot_numerical(labels: tuple[int], values: tuple[int], ax: Axes, *, title: str = "") -> None:
    """Plots barplot for numerical values.

    Args:
        labels: X-coordinates of the values.
        values: Values corresponding to each label.
        ax: On this axis the plot is created.
        title: The axis title, which is optional. Defaults to None.
    """
    ax.bar(labels, values)
    ax.set_ylabel("Frequency")
    ax.set_title(title)


def set_fig_title(fig: Figure, title: str, source_language: str, target_language: str) -> None:
    """Sets the title of the specified figure.

    Args:
        fig: Figure whose title is set.
        title: Title of the figure.
        source_language (str): Used to determine corpus name.
        target_language (str): Used to determine corpus name.
    """
    corpus_name = LONG_FORM[source_language] + "-" + LONG_FORM[target_language] + " Parallel Corpus"
    fig.suptitle(f"{title} in {corpus_name}")


def empty_counter_warning(mode: str) -> None:
    """Creates a warning for the specified mode that a counter is empty.

    Args:
        mode: The mode for which the warning is created.
    """
    warning = (
        f"Mode: {mode}: A counter seems to be empty. "
        "Perhaps a threshold value is set too high. "
        f"The creation of the {mode} plot is ommited."
    )
    warnings.warn(warning, stacklevel=2)


def savefig_(filename: str, fig: Figure) -> None:
    """Saves the plot under the specified file name and closes the figure.

    Args:
        filename: File name under which the plot is saved. The prefered extension is appended automatically.
        fig: Figure of the plot, which is closed after the plot is saved.
    """
    plt.savefig(PLOTS_PATH / f"{filename + PLOT_EXTENSION}")
    plt.close(fig)


def save_raw_counter(filename: str, counter: Counter[Any]) -> None:
    """Saves the input counter as a json file.

    Args:
        filename: Name of the json file.
        counter: The counter to be saved.
    """
    with (RAW_COUNTERS_PATH / filename).open("w") as json_file:
        json.dump(dict(counter), json_file)


def sentence_length_difference(
    corpus: EuroparlCorpus | DataPoint | Sequence[DataPoint],
    source_language: str,
    target_language: str,
    threshold: int = 1,
) -> None:
    """Plots the distribution of the sentence length difference between source and target language sentences.

    Args:
        corpus: EuroparlCorpus, a EuroparlCorpus slice or a EuroparlCorpus data point.
        source_language: Souce language.
        target_language: Target language.
        threshold: Counts below the threshold are discarded. Defaults to 1.
    """
    counter: Counter[int] = Counter()
    _fill_counters(
        counter,
        corpus,
        source_language,
        target_language,
        mode="sentence_lenth_difference",
    )

    filtered_counter = apply_threshold(counter, threshold=threshold)
    items = filtered_counter.items()
    if not items:
        empty_counter_warning("sentence_lenth_difference")
        return
    labels, values = zip(*items, strict=False)
    fig, ax = plt.subplots()
    _barplot_numerical(labels, values, ax)
    plt.axvline(fmean(filtered_counter.elements()), color="k", linestyle="dashed")
    abs_max = max(abs(max(labels)), abs(min(labels)))
    limit = abs_max - abs_max % 5
    plt.xticks(np.arange(-limit, limit + 1, 5))
    plt.xlabel(f"Difference ({source_language}-{target_language})")

    set_fig_title(fig, "Sentence length difference distribution", source_language, target_language)
    fig.tight_layout()
    savefig_(f"sentence_length_difference_{threshold}_{source_language}_{target_language}", fig)


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
    tagger_source = load_spacy_tagger(source_language)
    tagger_target = load_spacy_tagger(target_language)
    counters: dict[str, Counter[int]] = {source_language: Counter(), target_language: Counter()}
    _fill_counters(
        counters,
        corpus,
        source_language,
        target_language,
        mode="pos_tags",
        tagger_source=tagger_source,
        tagger_target=tagger_target,
    )

    fig, _ = plt.subplots(nrows=2, ncols=1, sharey=True)

    for i, language in enumerate([source_language, target_language]):
        items = counters[language].most_common(k)
        labels, values = zip(*items, strict=False)
        ax = fig.get_axes()[i]
        _barplot_categorical(labels, values, ax, title=f"{LONG_FORM[language]}")

    set_fig_title(fig, f"Top-{k} Most Frequent POS-Tags", source_language, target_language)
    fig.tight_layout()
    savefig_(f"top_{k}_pos_tags_{source_language}_{target_language}", fig)


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
    _fill_counters(
        counters,
        corpus,
        source_language,
        target_language,
        mode=cast(MODES, f"{mode}_length"),
    )

    fig, _ = plt.subplots(nrows=2, ncols=1, sharey=True, sharex=True)
    max_value = -1
    for i, language in enumerate([source_language, target_language]):
        items = apply_threshold(counters[language], threshold=threshold).items()
        if not items:
            empty_counter_warning(f"{mode}_length")
            plt.close(fig)
            return
        labels, values = zip(*items, strict=False)
        ax = fig.get_axes()[i]
        _barplot_numerical(labels, values, ax, title=f"{LONG_FORM[language]}")
        max_value = max(max_value, *labels)

    step = 1 if mode == "word" else 10
    start = 1 if mode == "word" else 0
    plt.xticks(np.arange(start, max_value + 1, step))
    plt.xlabel(f"{mode.capitalize()} Length")

    set_fig_title(fig, f"{mode.capitalize()} Length Distribution", source_language, target_language)
    fig.tight_layout()
    savefig_(f"{mode}_length_{threshold}_{source_language}_{target_language}", fig)


def most_frequent_words(
    corpus: EuroparlCorpus | DataPoint | Sequence[DataPoint],
    source_language: str,
    target_language: str,
    k: int = 20,
) -> None:
    """Plots the k most frequent words for each language.

    Args:
        corpus: EuroparlCorpus, a EuroparlCorpus slice or a EuroparlCorpus data point.
        source_language: Source language.
        target_language: Target language.
        k: The k most frequent words are plotted. Defaults to 20.
    """
    counters: dict[str, Counter[str]] = {source_language: Counter(), target_language: Counter()}
    _fill_counters(counters, corpus, source_language, target_language, mode="word")

    fig, _ = plt.subplots(nrows=2, ncols=1, sharey=True)

    for i, language in enumerate([source_language, target_language]):
        items = remove_punctuation(counters[language]).most_common(k)
        labels, values = zip(*items, strict=False)
        ax = fig.get_axes()[i]
        _barplot_categorical(labels, values, ax, title=f"{LONG_FORM[language]}")

    set_fig_title(fig, f"Top-{k} Most Frequent Words", source_language, target_language)
    fig.tight_layout()
    savefig_(f"top_{k}_most_frequent_{source_language}_{target_language}", fig)


def main() -> None:
    random.seed(42)
    corpus: EuroparlCorpus = EuroparlCorpus.load(source_language="en", target_language="de")

    most_frequent_words(corpus, corpus.source_language, corpus.target_language, k=15)
    length_(corpus, corpus.source_language, corpus.target_language, mode="word", threshold=100000)
    length_(corpus, corpus.source_language, corpus.target_language, mode="sentence", threshold=310)
    part_of_speech_tags(random.sample(corpus, 100000), corpus.source_language, corpus.target_language, k=10)
    sentence_length_difference(corpus, corpus.source_language, corpus.target_language, threshold=500)


if __name__ == "__main__":
    PLOTS_PATH.mkdir(parents=True, exist_ok=True)
    RAW_COUNTERS_PATH.mkdir(parents=True, exist_ok=True)
    main()
