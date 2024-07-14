import shutil
import urllib.parse
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Final, NamedTuple, Self, overload

import translator
from translator.utils.download import download_from_url


class DataPoint(NamedTuple):
    source_sentence: str
    target_sentence: str


class ParallelCorpus(Sequence[DataPoint]):
    def __init__(
        self,
        source_sentences: Sequence[str],
        target_sentences: Sequence[str],
        source_language: str,
        target_language: str,
    ) -> None:
        if len(source_sentences) != len(target_sentences):
            msg: str = "The parallel corpus requires the same number of source and target sentences."
            raise ValueError(msg)

        self.source_sentences = source_sentences
        self.target_sentences = target_sentences
        self.source_language = source_language
        self.target_language = target_language

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
