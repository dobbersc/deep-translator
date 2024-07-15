import functools
import spacy
from spacy.language import Language

@functools.lru_cache(maxsize=1)
def load_english_spacy_tagger() -> Language:
    """Loads the english spacy pipeline and excludes some not needed components.

    Returns:
        English spacy pipeline for POS-tagging.
    """
    return spacy.load(
        "en_core_web_sm",
        exclude=("parser", "senter", "attribute_ruler", "lemmatizer", "ner"),
    )


@functools.lru_cache(maxsize=1)
def load_german_spacy_tagger() -> Language:
    """Loads the german spacy pipeline and excludes some not needed components.

    Returns:
        German spacy pipeline for POS-tagging.
    """
    return spacy.load(
        "de_core_news_sm",
        exclude=("morphologizer", "parser", "lemmatizer", "senter", "attribute_ruler", "ner"),
    )

def load_spacy_tagger(language: str) -> Language:

    """Loads the spacy pipeline corresponding to the language provided.

    Args:
        language: Determines the spacy pipeline that is loaded.

    Raises:
        ValueError: In case the specified language is not supported.

    Returns:
        Spacy pipeline for POS-tagging.
    """
    if language == "en":
        return load_english_spacy_tagger()
    if language == "de":
        return load_german_spacy_tagger()
    msg = f"Language {language} is not supported."
    raise ValueError(msg)