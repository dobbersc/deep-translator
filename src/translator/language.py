from collections import Counter
from collections.abc import Iterable, Sized
from typing import Self


def _validate_token(token: str) -> None:
    reserved_token_to_error_message: dict[str, str] = {
        "<PAD>": "The reserved token '<PAD>' for padding sequences of unequal lengths exists inside the provided data.",
        "<START>": "The reserved token '<START>' to mark the source's start exists inside the provided data.",
        "<SEP>": "The reserved token '<SEP>' to separate source from target exists inside the provided data.",
        "<STOP>": "The reserved token '<STOP>' to mark the target's end exists inside the provided data.",
        "<UNK>": "The reserved token '<UNK>' for out-of-vocabulary words exists inside the provided data.",
    }
    if (msg := reserved_token_to_error_message.get(token)) is not None:
        raise ValueError(msg)


def build_language_dictionary(sentences: Iterable[Iterable[str]], unk_threshold: int = 0) -> dict[str, int]:
    """Builds a dictionary mapping tokens to an index value from tokenized sentences.

    Args:
        sentences: The list of tokenized sentences.
        unk_threshold: An integer threshold
            where all words that occur under this threshold will be excluded from the dictionary.

    Returns:
        The language token-to-index dictionary.
    """
    token2idx: dict[str, int] = {"<PAD>": 0, "<START>": 1, "<SEP>": 2, "<STOP>": 3, "<UNK>": 4}

    token_counter: Counter[str] = Counter(token for sentence in sentences for token in sentence)
    for token, count in token_counter.items():
        _validate_token(token)
        if count >= unk_threshold:
            token2idx[token] = len(token2idx)

    return token2idx


class Language(Sized):
    """Abstracts a language as a token-to-index (and reverse) dictionary."""

    def __init__(
        self,
        name: str,
        token2idx: dict[str, int],
        padding_token: str = "<PAD>",
        start_token: str = "<START>",
        seperator_token: str = "<SEP>",
        stop_token: str = "<STOP>",
        unknown_token: str = "<UNK>",
    ) -> None:
        """Initializes a Language.

        Args:
            name: The language's name.
            token2idx: The language's token-to-index dictionary.
            padding_token: The special token for padding sequences of unequal lengths exists.
            start_token: The special token to mark the source's start.
            seperator_token: The special token to separate source from target.
            stop_token: The special token to mark the target's end.
            unknown_token: The special token for out-of-vocabulary words.
        """
        self.name = name
        self.token2idx: dict[str, int] = token2idx
        self.idx2token: dict[int, str] = {value: key for key, value in token2idx.items()}

        special_tokens: set[str] = {padding_token, start_token, seperator_token, stop_token, unknown_token}
        if missing_special_tokens := special_tokens.difference(token2idx.keys()):
            msg: str = (
                f"The special tokens {', '.join(repr(t) for t in sorted(missing_special_tokens))} "
                f"are not present in the word-to-index dictionary."
            )
            raise ValueError(msg)

        self.padding_token = padding_token
        self.start_token = start_token
        self.seperator_token = seperator_token
        self.stop_token = stop_token
        self.unknown_token = unknown_token

    def get_index(self, token: str) -> int:
        """Gets the corresponding index to the provided token.

        If the token is not part of the language, the index of the "unknown token" will be returned.

        Args:
            token: The query token.

        Returns:
            The token's index.
        """
        return self.token2idx.get(token, self.unknown_token_index)

    def get_token(self, index: int) -> str:
        """Gets the corresponding token to the provided index.

        Args:
            index: The query index.

        Returns:
            The index's token.
        """
        return self.idx2token[index]

    def encode(self, tokens: Iterable[str]) -> list[int]:
        """Encodes tokens to their index representation.

        Args:
            tokens: An iterable of tokens.

        Returns:
            An list of indices corresponding to the input tokens.
        """
        return [self.get_index(token) for token in tokens]

    def decode(self, indices: Iterable[int]) -> list[str]:
        """Decodes indices to their token representation.

        Args:
            indices: An iterable of indices.

        Returns:
            An list of tokens corresponding to the input indices.
        """
        return [self.get_token(index) for index in indices]

    @property
    def padding_token_index(self) -> int:
        return self.token2idx[self.padding_token]

    @property
    def start_token_index(self) -> int:
        return self.token2idx[self.start_token]

    @property
    def seperator_token_index(self) -> int:
        return self.token2idx[self.seperator_token]

    @property
    def stop_token_index(self) -> int:
        return self.token2idx[self.stop_token]

    @property
    def unknown_token_index(self) -> int:
        return self.token2idx[self.unknown_token]

    @property
    def vocabulary_size(self) -> int:
        return len(self)

    def __len__(self) -> int:
        return len(self.token2idx)

    def __repr__(self) -> str:
        return f"{type(self).__name__}<{len(self)}>(name={self.name!r})"

    @classmethod
    def from_sentences(
        cls,
        name: str,
        sentences: Iterable[Iterable[str]],
        unk_threshold: int = 0,
    ) -> Self:
        """Initializes a language from tokenized sentences.

        Args:
            name: The name of the language.
            sentences: A list of tokenized sentences in the target language.
            unk_threshold: An integer threshold
                where all words that occur under this threshold will be excluded from the dictionary.

        Returns:
            An instance of the Language.
        """
        return cls(name=name, token2idx=build_language_dictionary(sentences, unk_threshold=unk_threshold))
