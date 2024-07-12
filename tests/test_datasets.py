import pytest

from translator.corpus import DataPoint, ParallelCorpus


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
