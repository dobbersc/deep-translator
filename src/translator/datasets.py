import shutil
import urllib.parse
import urllib.request
from collections.abc import Callable, Sequence, Sized
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Final, NamedTuple, Self, overload

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

import translator
from translator.language import Language
from translator.preprocessing import Tokenizer
from translator.utils.download import download_from_url


class DataPoint(NamedTuple):
    """Encapsulated the source and target for machine translation."""

    source: str
    target: str


class VectorizedDataPoint(NamedTuple):
    """Encapsulated the vectorized source and target for machine translation."""

    source: Tensor
    target: Tensor


class VectorizedDataPointBatch(Sequence[VectorizedDataPoint]):
    def __init__(self, *data_points: VectorizedDataPoint, source_padding_value: int, target_padding_value: int) -> None:
        if not data_points:
            msg: str = "Received empty batch of data points."
            raise ValueError(msg)

        self._data_points = data_points
        self.source_padding_value = source_padding_value
        self.target_padding_value = target_padding_value

        sources, targets = zip(*self._data_points, strict=True)
        self.sources: Tensor = torch.nn.utils.rnn.pad_sequence(
            list(sources),
            batch_first=True,
            padding_value=source_padding_value,
        )
        self.targets: Tensor = torch.nn.utils.rnn.pad_sequence(
            list(targets),
            batch_first=True,
            padding_value=target_padding_value,
        )

    @overload
    def __getitem__(self, index: int) -> VectorizedDataPoint:
        ...

    @overload
    def __getitem__(self, index: slice) -> Self:
        ...

    def __getitem__(self, index: int | slice) -> VectorizedDataPoint | Self:
        if isinstance(index, int):
            return self._data_points[index]
        return type(self)(
            *self._data_points[index],
            source_padding_value=self.source_padding_value,
            target_padding_value=self.target_padding_value,
        )

    def __len__(self) -> int:
        return len(self._data_points)


class VectorizedParallelDataset(Dataset[VectorizedDataPoint], Sized):
    def __init__(
        self,
        source_sentences: Sequence[str],
        target_sentences: Sequence[str],
        source_tokenizer: Tokenizer,
        target_tokenizer: Tokenizer,
        source_language: Language,
        target_language: Language,
    ) -> None:
        if len(source_sentences) != len(target_sentences):
            msg: str = f"The {type(self).__name__} requires the same number of source and target sentences."
            raise ValueError(msg)

        self.source_sentences = source_sentences
        self.target_sentences = target_sentences

        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer

        self.source_language = source_language
        self.target_language = target_language

    def __getitem__(self, index: int) -> VectorizedDataPoint:
        source: str = self.source_sentences[index]
        target: str = self.target_sentences[index]
        return VectorizedDataPoint(
            source=torch.tensor(self.source_language.encode(self.source_tokenizer(source)), dtype=torch.long),
            target=torch.tensor(self.target_language.encode(self.target_tokenizer(target)), dtype=torch.long),
        )

    def __len__(self) -> int:
        return len(self.source_sentences)


class ParallelDataLoader(DataLoader[VectorizedDataPoint]):
    def __init__(
        self,
        dataset: VectorizedParallelDataset,
        *args: Any,
        collate_fn: Callable[[list[VectorizedDataPoint]], VectorizedDataPointBatch] | None = None,
        **kwargs: Any,
    ) -> None:
        source_padding_value = dataset.source_language.padding_token_index
        target_padding_value = dataset.target_language.padding_token_index
        if collate_fn is None:
            kwargs["collate_fn"] = lambda batch: VectorizedDataPointBatch(
                *batch,
                source_padding_value=source_padding_value,
                target_padding_value=target_padding_value,
            )
        else:
            kwargs["collate_fn"] = collate_fn
        super().__init__(dataset, *args, **kwargs)


class ParallelCorpus(Sequence[DataPoint]):
    def __init__(
        self,
        source_sentences: Sequence[str],
        target_sentences: Sequence[str],
        source_language: str,
        target_language: str,
    ) -> None:
        if len(source_sentences) != len(target_sentences):
            msg: str = f"The {type(self).__name__} requires the same number of source and target sentences."
            raise ValueError(msg)

        self.source_sentences = source_sentences
        self.target_sentences = target_sentences
        self.source_language = source_language
        self.target_language = target_language

    def downsample(self, size: float, random_state: int | np.random.RandomState | None = 42) -> Self:
        sources, _, targets, _ = train_test_split(
            self.source_sentences,
            self.target_sentences,
            train_size=size,
            shuffle=True,
            random_state=random_state,
        )
        return type(self)(sources, targets, self.source_language, self.target_language)

    def split(
        self,
        train_size: float = 0.7,
        dev_size: float = 0.1,
        test_size: float = 0.2,
        random_state: int | np.random.RandomState | None = 42,
    ) -> tuple[Self, Self, Self]:
        source_train_and_dev, source_test, target_train_and_dev, target_test = train_test_split(
            self.source_sentences,
            self.target_sentences,
            test_size=test_size,
            shuffle=True,
            random_state=random_state,
        )
        source_train, source_dev, target_train, target_dev = train_test_split(
            source_train_and_dev,
            target_train_and_dev,
            train_size=train_size / (train_size + dev_size),
            shuffle=True,
            random_state=random_state,
        )

        languages: tuple[str, str] = (self.source_language, self.target_language)
        return (
            type(self)(source_train, target_train, *languages),
            type(self)(source_dev, target_dev, *languages),
            type(self)(source_test, target_test, *languages),
        )

    @overload
    def __getitem__(self, index: int) -> DataPoint:
        ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[DataPoint]:
        ...

    def __getitem__(self, index: int | slice) -> DataPoint | Sequence[DataPoint]:
        if isinstance(index, int):
            return DataPoint(self.source_sentences[index], self.target_sentences[index])
        return tuple(
            DataPoint(source_sentence, target_sentence)
            for source_sentence, target_sentence in zip(
                self.source_sentences[index],
                self.target_sentences[index],
                strict=True,
            )
        )

    def __len__(self) -> int:
        return len(self.source_sentences)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}<{len(self)}>("
            f"source_language={self.source_language!r}, "
            f"target_language={self.target_language!r})"
        )


class EuroparlCorpus(ParallelCorpus):
    """Europarl Parallel Corpus.

    Source: https://www.statmt.org/europarl/
    Reference: "Europarl: A Parallel Corpus for Statistical Machine Translation, Philipp Koehn, MT Summit 2005"
    """

    # Specifies the valid language alignments available on the Europarl server (https://www.statmt.org/europarl/v7)
    VALID_LANGUAGES: Final[frozenset[tuple[str, str]]] = frozenset(
        (target_language, "en")
        for target_language in (
            "bg",  # Bulgarian-English, 41 MB, 01/2007-11/2011
            "cs",  # Czech-English, 60 MB, 01/2007-11/2011
            "da",  # Danish-English, 179 MB, 04/1996-11/2011
            "de",  # German-English, 189 MB, 04/1996-11/2011
            "el",  # Greek-English, 145 MB, 04/1996-11/2011
            "es",  # Spanish-English, 187 MB, 04/1996-11/2011
            "et",  # Estonian-English, 57 MB, 01/2007-11/2011
            "fi",  # Finnish-English, 179 MB, 01/1997-11/2011
            "fr",  # French-English, 194 MB, 04/1996-11/2011
            "hu",  # Hungarian-English, 59 MB, 01/2007-11/2011
            "it",  # Italian-English, 188 MB, 04/1996-11/2011
            "lt",  # Lithuanian-English, 57 MB, 01/2007-11/2011
            "lv",  # Latvian-English, 57 MB, 01/2007-11/2011
            "nl",  # Dutch-English, 190 MB, 04/1996-11/2011
            "pl",  # Polish-English, 59 MB, 01/2007-11/2011
            "pt",  # Portuguese-English, 189 MB, 04/1996-11/2011
            "ro",  # Romanian-English, 37 MB, 01/2007-11/2011
            "sk",  # Slovak-English, 59 MB, 01/2007-11/2011
            "sl",  # Slovene-English, 54 MB, 01/2007-11/2011
            "sv",  # Swedish-English, 171 MB, 01/1997-11/2011
        )
    )

    @staticmethod
    def _preprocess(
        source_sentences: Sequence[str],
        target_sentences: Sequence[str],
    ) -> tuple[Sequence[str], Sequence[str]]:
        """Pre-processes the dataset's source and target sentences.

        This function applies the following pre-processing steps:

        - Removes data points with an empty source or target sentence.
        - TODO: ...

        Args:
            source_sentences: A sequence of source sentences.
            target_sentences: A sequence of target sentences.

        Returns:
            The pre-processed source and target sentences.
        """
        processed_source_sentences: list[str] = []
        processed_target_sentences: list[str] = []

        for source_sentence, target_sentence in zip(source_sentences, target_sentences, strict=True):
            if not source_sentence or not target_sentence:
                continue

            processed_source_sentences.append(source_sentence)
            processed_target_sentences.append(target_sentence)

        return processed_source_sentences, processed_target_sentences

    @classmethod
    def from_files(
        cls,
        source_language_file: Path,
        target_language_file: Path,
        source_language: str,
        target_language: str,
    ) -> Self:
        source_sentences: list[str] = source_language_file.read_text().splitlines()
        target_sentences: list[str] = target_language_file.read_text().splitlines()
        return cls(*cls._preprocess(source_sentences, target_sentences), source_language, target_language)

    @classmethod
    def load(
        cls,
        source_language: str,
        target_language: str,
        cache_directory: Path = translator.CACHE_DIRECTORY,
        url: str = "https://www.statmt.org/europarl/v7/",
        version: str = "v7",
    ) -> Self:
        """Loads the Europarl parallel corpus from ISO language codes.

        Args:
            source_language: The language code of the source language (ISO 639 Set 1).
            target_language: The language code of the target language (ISO 639 Set 1).
            cache_directory: The directory where to cache the dataset files for future use.
                Per default, the package's cache directory will be used.
            url: The URL to the server where the corpus archives are located.
            version: The version of the corpus.

        Returns:
            The Europarl parallel corpus of the specified languages.
        """
        corpus_stem: str
        if (source_language, target_language) in cls.VALID_LANGUAGES:
            corpus_stem = f"{source_language}-{target_language}"
        elif (target_language, source_language) in cls.VALID_LANGUAGES:
            corpus_stem = f"{target_language}-{source_language}"
        else:
            msg: str = f"No parallel corpus was found from {source_language!r} to {target_language!r}."
            raise ValueError(msg)

        cache_directory = cache_directory / "datasets" / "europarl"
        source_language_path: Path = cache_directory / f"europarl-{version}.{corpus_stem}.{source_language}"
        target_language_path: Path = cache_directory / f"europarl-{version}.{corpus_stem}.{target_language}"

        if not (target_language_path.exists() and source_language_path.exists()):
            archive_file = f"{corpus_stem}.tgz"
            corpus_url: str = urllib.parse.urljoin(url, archive_file)

            with TemporaryDirectory() as tmpdir_:
                tmpdir = Path(tmpdir_)

                archive_path: Path = tmpdir / archive_file
                download_from_url(corpus_url, destination=archive_path)
                shutil.unpack_archive(archive_path, extract_dir=tmpdir)

                cache_directory.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(tmpdir / source_language_path.name, source_language_path)
                shutil.copyfile(tmpdir / target_language_path.name, target_language_path)

        return cls.from_files(source_language_path, target_language_path, source_language, target_language)
