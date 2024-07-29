import functools
import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from tqdm import tqdm

from translator.datasets import EuroparlCorpus
from translator.tokenizers import Tokenizer, preprocess
from translator.utils.random import set_seed

from experiments import RESULTS_PATH

if TYPE_CHECKING:
    from pathlib import Path

logger: logging.Logger = logging.getLogger("translator")


class Callback(CallbackAny2Vec):
    def __init__(self) -> None:
        super().__init__()
        self.epoch: int = 1
        self.loss_previous_step: float = 0

    def on_epoch_end(self, model: Word2Vec) -> None:
        loss: float = model.get_latest_training_loss()

        if self.epoch == 1:
            logger.info("Loss Epoch %d: %.4f", self.epoch, loss)
        else:
            logger.info("Loss Epoch %d: %.4f", self.epoch, loss - self.loss_previous_step)

        self.epoch += 1
        self.loss_previous_step = loss


def train(langauge: str, sentences: Sequence[str]) -> None:
    tokenizer: Tokenizer = functools.partial(preprocess, language=langauge, lowercase=True)

    logger.info("Training Embeddings for Language %r", langauge)
    tokenized_sentences: list[Sequence[str]] = [
        tokenizer(sentence) for sentence in tqdm(sentences, desc="Tokenizing Sentences", unit="sentence")
    ]
    pretrained_embeddings: Word2Vec = Word2Vec(
        sentences=tokenized_sentences,
        vector_size=300,
        min_count=5,
        epochs=10,
        compute_loss=True,
        workers=64,
        callbacks=[Callback()],
    )
    out_directory: Path = RESULTS_PATH / "embeddings" / langauge
    out_directory.mkdir(parents=True, exist_ok=True)
    pretrained_embeddings.wv.save_word2vec_format(out_directory / "vectors.txt")


def main(source_language: str = "de", target_language: str = "en") -> None:
    set_seed(42)

    corpus: EuroparlCorpus = EuroparlCorpus.load(source_language, target_language).downsample(0.1)
    train_split, _, _ = corpus.split()

    train(train_split.source_language, train_split.source_sentences)
    train(train_split.target_language, train_split.target_sentences)


if __name__ == "__main__":
    main()
