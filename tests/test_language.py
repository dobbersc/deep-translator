import numpy as np
import pytest
from gensim.models import KeyedVectors

from translator.language import Language


class TestLanguage:
    def test_language_from_sentences(self) -> None:
        sentences: tuple[list[str], ...] = (
            "Dies ist ein Beispielsatz .".split(),
            "Dies ist ein weiterer Beispielsatz .".split(),
        )
        language: Language = Language.from_sentences(name="de", sentences=sentences, unk_threshold=2)

        assert language.vocabulary_size == len(language) == 10

        assert language.get_index("Beispielsatz") == 8
        assert language.get_index("weiterer") == language.unknown_token_index
        assert language.get_token(5) == "Dies"

        assert language.encode(("Ein", "Beispielsatz", ".")) == [language.unknown_token_index, 8, 9]
        assert language.decode((language.unknown_token_index, 8, 9)) == ["<UNK>", "Beispielsatz", "."]

        assert language.padding_token_index == 0
        assert language.start_token_index == 1
        assert language.seperator_token_index == 2
        assert language.stop_token_index == 3
        assert language.unknown_token_index == 4

    def test_language_from_sentences_with_special_tokens_in_sentences(self) -> None:
        sentences: tuple[list[str], ...] = (
            "<START> Dies ist ein Beispielsatz . <STOP>".split(),
            "<START> Dies ist ein weiterer Beispielsatz . <STOP>".split(),
        )
        with pytest.raises(ValueError, match="The reserved token '<START>' .+ exists inside the provided data."):
            Language.from_sentences(name="de", sentences=sentences)

    def test_from_embeddings(self) -> None:
        embeddings: KeyedVectors = KeyedVectors(vector_size=10, count=1)
        embeddings.add_vector("Test", np.ones(10))

        language: Language = Language.from_embeddings("de", embeddings)

        assert (
            language.token2idx
            == embeddings.key_to_index
            == {"Test": 0, "<PAD>": 1, "<START>": 2, "<SEP>": 3, "<STOP>": 4, "<UNK>": 5}
        )

    def test_invalid_special_tokens(self) -> None:
        msg: str = "The special tokens '<SEP>', '<START>', '<STOP>' are not present in the word-to-index dictionary."
        with pytest.raises(ValueError, match=msg):
            Language("de", {"<PAD>": 0, "<UNK>": 1})
