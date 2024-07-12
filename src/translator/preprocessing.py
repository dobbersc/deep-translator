import functools
import unicodedata
from typing import Final, Literal

import regex
from sacremoses import MosesTokenizer

__CONTROL_CHARACTERS_EXCEPT_TAB_CR_LF: Final[str] = "[^\\P{C}\t\r\n]"
__PROBLEM_CHARACTERS_PATTERN: Final[regex.Pattern[str]] = regex.compile(__CONTROL_CHARACTERS_EXCEPT_TAB_CR_LF)


def remove_problem_characters(text: str) -> str:
    return __PROBLEM_CHARACTERS_PATTERN.sub("", text)


@functools.cache
def load_moses_tokenizer(language: str) -> MosesTokenizer:
    return MosesTokenizer(language)


def preprocess(
    text: str,
    *,
    language: str = "en",
    unicode_normalization: Literal["NFD", "NFC", "NFKD", "NFKC"] | None = "NFKC",
    segmentation_level: Literal["word", "subword", "character"] = "word",
    lowercase: bool = False,
) -> list[str]:
    """Pre-processes the input text.

    Pre-processing steps:

    1. Remove problematic Unicode characters from the text, i.e. control characters except TAB, CR and LF.
    2. (Optional) Normalize the text to conform to the specified Unicode normalization form.
    3. Tokenize the text into semantic segments on the specified segmentation level.
    4. (Optional) Lowercase the text.

    Args:
        text: The text to pre-process.
        language: The language of the text. (Only used for word-level segmentation.)
        segmentation_level: The segmentation level of the text tokenization.
        unicode_normalization: Optional Unicode normalization form to normalize the text to.
            If None, no normalization will be applied.
        lowercase: If true, the text will be transformed to lowercase. Otherwise, the original casing will be preserved.

    Returns:
        The pre-processed tokenized text.
    """
    # TODO: Maybe integrate MosesPunctNormalizer

    text = remove_problem_characters(text)
    if unicode_normalization is not None:
        text = unicodedata.normalize(unicode_normalization, text)

    tokens: list[str]
    if segmentation_level == "word":
        tokenizer: MosesTokenizer = load_moses_tokenizer(language)
        tokens = tokenizer.tokenize(text)
    elif segmentation_level == "subword":
        raise NotImplementedError
    elif segmentation_level == "character":
        tokens = list(text)
    else:
        msg: str = f"Invalid segmentation level {segmentation_level!r}. Choose from 'word', 'subword' or 'character'."  # type: ignore[unreachable]
        raise ValueError(msg)

    if lowercase:
        tokens = [token.lower() for token in tokens]

    return tokens
