from pathlib import Path

import pytest
import torch

from translator.datasets import (
    DataPoint,
    EuroparlCorpus,
    ParallelCorpus,
    ParallelDataLoader,
    VectorizedDataPointBatch,
    VectorizedParallelDataset,
)
from translator.language import Language


@pytest.fixture()
def vectorized_dataset() -> VectorizedParallelDataset:
    source_sentences: list[str] = [
        "Dies ist ein Beispielsatz .",
        "Dies ist ein weiterer Beispielsatz .",
    ]
    target_sentences: list[str] = [
        "This is an example sentence .",
        "This is another example sentence .",
    ]
    return VectorizedParallelDataset(
        source_sentences=source_sentences,
        target_sentences=target_sentences,
        source_tokenizer=lambda x: x.split(),
        target_tokenizer=lambda x: x.split(),
        source_language=Language.from_sentences("de", map(str.split, source_sentences)),
        target_language=Language.from_sentences("en", map(str.split, target_sentences)),
    )


class TestVectorizedParallelDataset:
    def test_valid_dataset(self, vectorized_dataset: VectorizedParallelDataset) -> None:
        assert len(vectorized_dataset) == 2
        assert (vectorized_dataset[0].source == torch.tensor((5, 6, 7, 8, 9), dtype=torch.long)).all()
        assert (vectorized_dataset[0].target == torch.tensor((5, 6, 7, 8, 9, 10), dtype=torch.long)).all()

    def test_invalid_source_and_target_sentences(self) -> None:
        with pytest.raises(ValueError, match="requires the same number of source and target sentences."):
            VectorizedParallelDataset(
                source_sentences=["Satz 1", "Satz 2"],
                target_sentences=["Sentence 1", "Sentence 2", "Sentence 3"],
                source_tokenizer=lambda x: x.split(),
                target_tokenizer=lambda x: x.split(),
                source_language=Language.from_sentences("de", []),
                target_language=Language.from_sentences("en", []),
            )


def test_parallel_data_loader(vectorized_dataset: VectorizedParallelDataset) -> None:
    data_loader: ParallelDataLoader = ParallelDataLoader(vectorized_dataset, batch_size=2)
    assert len(data_loader) == 1

    batch: VectorizedDataPointBatch = next(iter(data_loader))
    assert len(batch) == 2
    assert (batch.sources == torch.tensor(((5, 6, 7, 8, 9, 0), (5, 6, 7, 10, 8, 9)), dtype=torch.long)).all()
    assert (batch.targets == torch.tensor(((5, 6, 7, 8, 9, 10), (5, 6, 11, 8, 9, 10)), dtype=torch.long)).all()


class TestParallelCorpus:
    def test_valid_corpus(self) -> None:
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
        with pytest.raises(ValueError, match="requires the same number of source and target sentences."):
            ParallelCorpus(
                source_sentences=["Satz 1", "Satz 2"],
                target_sentences=["Sentence 1", "Sentence 2", "Sentence 3"],
                source_language="de",
                target_language="en",
            )

    def test_downsample(self) -> None:
        corpus: ParallelCorpus = ParallelCorpus(
            source_sentences=[f"Satz {i}" for i in range(100)],
            target_sentences=[f"Sentence {i}" for i in range(100)],
            source_language="de",
            target_language="en",
        )

        downsampled: ParallelCorpus = corpus.downsample(0.1)
        assert len(downsampled) == 10

    def test_split(self) -> None:
        corpus: ParallelCorpus = ParallelCorpus(
            source_sentences=[f"Satz {i}" for i in range(100)],
            target_sentences=[f"Sentence {i}" for i in range(100)],
            source_language="de",
            target_language="en",
        )

        train, dev, test = corpus.split(0.7, 0.1, 0.2)
        assert len(train) == 70
        assert len(dev) == 10
        assert len(test) == 20


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
        assert len(corpus) == len(reverse_corpus) == 1908920  # Before pre-processing: 1920209

        # Check if cached data has been used
        assert de_language_path.stat().st_mtime == de_last_modified
        assert en_language_path.stat().st_mtime == en_last_modified

    def test_load_invalid_corpus(self) -> None:
        with pytest.raises(ValueError, match="No parallel corpus was found"):
            EuroparlCorpus.load("INVALID", "en")
