from collections.abc import Sequence
from typing import Protocol

from torch import Tensor

from translator.language import Language
from translator.nn import DecoderLSTM, EncoderLSTM, Seq2Seq
from translator.preprocessing import Tokenizer


class Detokenizer(Protocol):
    def __call__(self, tokens: Sequence[str]) -> str:
        ...


class Translator(Seq2Seq):
    # Implement searches and complete pipeline here.
    # This should be the easy-to-use entry point

    def __init__(
        self,
        source_language: Language,
        target_language: Language,
        source_tokenizer: Tokenizer,
        target_detokenizer: Detokenizer = lambda x: " ".join(x),
        source_embedding_size: int = 256,
        target_embedding_size: int = 256,
        attention: None = None,  # noqa: ARG002
        hidden_size: int = 256,
    ) -> None:
        super().__init__(
            EncoderLSTM(
                vocabulary_size=source_language.vocabulary_size,
                embedding_size=source_embedding_size,
                hidden_size=hidden_size,
                padding_index=source_language.padding_token_index,
                start_index=source_language.start_token_index,
            ),
            DecoderLSTM(
                vocabulary_size=target_language.vocabulary_size,
                embedding_size=target_embedding_size,
                hidden_size=hidden_size,
                padding_index=target_language.padding_token_index,
                seperator_index=target_language.seperator_token_index,
                stop_index=target_language.stop_token_index,
            ),
        )

        self.source_language = source_language
        self.target_language = target_language
        self.tokenizer = source_tokenizer
        self.detokenizer = target_detokenizer

    def forward(self, sources: Tensor, targets: Tensor, teacher_forcing_ratio: float = 0.5) -> Tensor:
        # returns the log probabilities for each word in the source sequences (will later from the seq2seq model)
        # to calculate the cross entropy you need the NLLLoss from pytorch
        # or see here for perplexity https://huggingface.co/docs/transformers/en/perplexity
        raise NotImplementedError

    def translate(self, text: str) -> str:
        raise NotImplementedError

    def evaluate_perplexity(self, sources: Sequence[str], targets: Sequence[str], batch_size: int = 32) -> float:
        # use forward with teacher_forcing_ratio=0
        # can also leave batch size out for now
        raise NotImplementedError

    def evaluate_bleu(self, sources: Sequence[str], targets: Sequence[str], batch_size: int = 32) -> float:
        raise NotImplementedError
