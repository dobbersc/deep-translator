from pathlib import Path

import pytest

from translator.datasets import DataPoint, EuroparlCorpus, ParallelCorpus


class TestParallelCorpus:
    def test_properties(
        self,
    ) -> None:
        corpus: ParallelCorpus = ParallelCorpus(
            source_sentences=["Satz 1", "Satz 2", "Satz 3"],
            target_sentences=["Sentence 1", "Sentence 2", "Sentence 3"],
            source_language="de",
            target_language="en",
        )

        assert len(corpus) == 3
        assert list(corpus) == [
            DataPoint("Satz 1", "Sentence 1"),
            DataPoint("Satz 2", "Sentence 2"),
            DataPoint("Satz 3", "Sentence 3"),
        ]
        assert corpus[:2] == (DataPoint("Satz 1", "Sentence 1"), DataPoint("Satz 2", "Sentence 2"))

    def test_invalid_source_and_target_sentences(self) -> None:
        with pytest.raises(
            ValueError,
            match="The parallel corpus requires the same number of source and target sentences.",
        ):
            ParallelCorpus(
                source_sentences=["Satz 1", "Satz2"],
                target_sentences=["Sentence 1", "Sentence 2", "Sentence 3"],
                source_language="de",
                target_language="en",
            )


class TestEuroparlCorpus:
    def test_load_valid_corpus(self, tmp_path: Path) -> None:
        cache_directory: Path = tmp_path / "datasets" / "europarl"
        de_language_path: Path = cache_directory / "europarl-v7.de-en.de"
        en_language_path: Path = cache_directory / "europarl-v7.de-en.en"

        # Load de -> en parallel corpus
        corpus: EuroparlCorpus = EuroparlCorpus.load("de", "en", cache_directory=tmp_path)
        assert de_language_path.exists()
        assert en_language_path.exists()

        de_last_modified: float = de_language_path.stat().st_mtime
        en_last_modified: float = en_language_path.stat().st_mtime

        # Load en -> de parallel corpus (cached)
        reverse_corpus: EuroparlCorpus = EuroparlCorpus.load("en", "de", cache_directory=tmp_path)
        assert len(corpus) == len(reverse_corpus) == 1920209

        # Check if cached data has been used
        assert de_language_path.stat().st_mtime == de_last_modified
        assert en_language_path.stat().st_mtime == en_last_modified

    def test_load_invalid_corpus(self) -> None:
        with pytest.raises(ValueError, match="No parallel corpus was found"):
            EuroparlCorpus.load("INVALID", "en")
