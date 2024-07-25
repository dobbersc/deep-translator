import math
import warnings
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import IO, Any, Literal, Self, cast, overload

import torch
from gensim.models import KeyedVectors
from sacrebleu.metrics.bleu import BLEU
from torch import Tensor
from tqdm import tqdm

from translator.datasets import VectorizedDataPointBatch
from translator.language import Language
from translator.nn import DecoderLSTM, EncoderLSTM, Seq2Seq
from translator.tokenizers import Detokenizer, Tokenizer


class Translator(Seq2Seq):
    def __init__(
        self,
        *,
        source_language: Language,
        target_language: Language,
        source_tokenizer: Tokenizer,
        target_tokenizer: Tokenizer,
        target_detokenizer: Detokenizer = " ".join,
        source_embedding_size: int = 256,
        target_embedding_size: int = 256,
        hidden_size: int = 512,
        attention: None = None,  # noqa: ARG002
        source_pretrained_embeddings: KeyedVectors | Tensor | None = None,
        target_pretrained_embeddings: KeyedVectors | Tensor | None = None,
        freeze_pretrained_embeddings: bool = True,
        embedding_dropout: float = 0,
        dropout: float = 0,
    ) -> None:
        super().__init__(
            EncoderLSTM(
                vocabulary_size=source_language.vocabulary_size,
                embedding_size=source_embedding_size,
                hidden_size=hidden_size,
                pretrained_embeddings=source_pretrained_embeddings,
                freeze_pretrained_embeddings=freeze_pretrained_embeddings,
                embedding_dropout=embedding_dropout,
                dropout=dropout,
                padding_index=source_language.padding_token_index,
                start_index=source_language.start_token_index,
            ),
            DecoderLSTM(
                vocabulary_size=target_language.vocabulary_size,
                embedding_size=target_embedding_size,
                hidden_size=hidden_size,
                pretrained_embeddings=target_pretrained_embeddings,
                freeze_pretrained_embeddings=freeze_pretrained_embeddings,
                embedding_dropout=embedding_dropout,
                dropout=dropout,
                padding_index=target_language.padding_token_index,
                seperator_index=target_language.seperator_token_index,
                stop_index=target_language.stop_token_index,
            ),
        )

        self.source_language = source_language
        self.target_language = target_language

        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer
        self.target_detokenizer = target_detokenizer

    def _truncate_predicted_targets(self, targets: Tensor) -> list[Tensor]:
        """Truncates the predicted target sequences before the stop token.

        This function processes a batch of predicted target sequences and
        truncates each sequence  before the first occurrence of the stop token.
        If a sequence does not contain the stop token, it remains unchanged.

        Args:
            targets: A tensor of token indices. Shape: [batch_size, sequence_length].

        Returns:
            A list of truncated target tensors, each with their individual sequence length.
            Shape: [effective_sequence_length].
        """
        stop_index_mask: Tensor = targets.eq(self.decoder.stop_index).to(targets.dtype)
        effective_sequence_lengths: Tensor = stop_index_mask.argmax(dim=1)
        effective_sequence_lengths[targets.ne(self.decoder.stop_index).all(dim=1)] = targets.size(dim=1)
        return [
            target.narrow(dim=0, start=0, length=sequence_length)
            for target, sequence_length in zip(targets, effective_sequence_lengths, strict=True)
        ]

    @overload
    def translate(
        self,
        texts: str,
        method: Literal["greedy", "sampled", "beam-search"] = ...,
        max_length: int = ...,
        **kwargs: Any,
    ) -> str:
        ...

    @overload
    def translate(
        self,
        texts: list[str],
        method: Literal["greedy", "sampled", "beam-search"] = ...,
        max_length: int = ...,
        **kwargs: Any,
    ) -> list[str]:
        ...

    @torch.no_grad()
    def translate(
        self,
        texts: str | list[str],
        method: Literal["greedy", "sampled", "beam-search"] = "sampled",
        max_length: int = 512,
        **kwargs: Any,
    ) -> str | list[str]:
        self.eval()

        is_batched_input: bool = isinstance(texts, list)
        if not is_batched_input:
            texts = cast(list[str], [texts])

        tokenized_sources: list[Sequence[str]] = [self.source_tokenizer(text) for text in texts]
        source_batch: list[Tensor] = [
            torch.tensor(self.source_language.encode(tokens), dtype=torch.long, device=self.device)
            for tokens in tokenized_sources
        ]
        sources: Tensor = torch.nn.utils.rnn.pad_sequence(
            source_batch,
            batch_first=True,
            padding_value=cast(int, self.encoder.padding_index),
        )

        targets: Tensor = self.generate(sources, method=method, max_length=max_length, **kwargs)

        translations: list[str] = []
        for target in self._truncate_predicted_targets(targets):
            tokenized_target: list[str] = self.target_language.decode(target)
            detokenized_target: str = self.target_detokenizer(tokenized_target)
            translations.append(detokenized_target)

        return translations if is_batched_input else translations[0]

    @torch.no_grad()
    def evaluate(
        self,
        data_points: Iterable[VectorizedDataPointBatch],
        criterion: Callable[[Tensor, Tensor], Tensor] | None = None,
    ) -> tuple[float, float]:
        # TODO: Support Iterable[DataPoint] and Iterable[VectorizedDataPoint]
        self.eval()  # Bring the model into evaluation mode.

        padding_index: int = -100 if self.decoder.padding_index is None else self.decoder.padding_index
        if criterion is None:
            criterion = torch.nn.NLLLoss(ignore_index=padding_index)

        nll_criterion: torch.nn.NLLLoss = torch.nn.NLLLoss(ignore_index=padding_index)

        # Keep track of running loss.
        validation_loss: float = 0
        validation_nll_loss: float = 0

        num_batches: int = 0
        for data_point_batch in tqdm(data_points, desc="Evaluating Model", unit="batch", leave=False):
            # Move data to the model's device.
            sources: Tensor = data_point_batch.sources.to(self.device)
            targets: Tensor = data_point_batch.targets.to(self.device)

            # With teacher forcing disabled, compute the log probabilities for each token in the target sequences.
            predicted_log_probabilities: Tensor = self(sources, targets, teacher_forcing_ratio=0).flatten(end_dim=1)
            ground_truth_targets: Tensor = self.decoder.make_ground_truth_sequences(targets).flatten()

            loss: Tensor = criterion(predicted_log_probabilities, ground_truth_targets)
            nll_loss: Tensor = nll_criterion(predicted_log_probabilities, ground_truth_targets)

            validation_loss += loss.item()
            validation_nll_loss += nll_loss.item()
            num_batches += 1

        validation_loss /= num_batches
        validation_nll_loss /= num_batches
        validation_perplexity: float = math.exp(validation_nll_loss)

        return validation_loss, validation_perplexity

    def evaluate_bleu(self, sources: Sequence[str], targets: Sequence[str], max_ngram_order: int = 4) -> dict[str, Any]:
        """Calculates the BLEU score for the targets and the translation of the given sources.

        Args:
            sources: Sources that are translated.
            targets: Target text for each of the given sources.
            max_ngram_order: Bleu considers all n-gram precisions, where 1<=n<=max_ngram_order. Defaults to 4.

        Returns:
            Dictionary containing the score, the individual precisions and the length of the translation and the target.
        """
        translated_sources: list[str] = [self.translate(source) for source in sources]
        bleu = BLEU(max_ngram_order=max_ngram_order)
        bleu_corpus_score = bleu.corpus_score(translated_sources, [targets])
        return {
            "score": bleu_corpus_score.score,
            "precisions": bleu_corpus_score.precisions,
            "source_len": bleu_corpus_score.sys_len,
            "target_len": bleu_corpus_score.ref_len,
        }

    def save(self, f: str | Path | IO[bytes]) -> None:
        if self.encoder.lstm.hidden_size != self.decoder.lstm.hidden_size:
            msg: str = "The hidden size of the encoder and decoder LSTMs must match to be serializable."
            raise AssertionError(msg)

        if self.encoder.dropout.p != self.decoder.dropout.p:
            warnings.warn(
                "Mismatch in embedding dropout probabilities between encoder and decoder embeddings. "
                "Saving the model with the encoder's dropout value for both. "
                f"Encoder dropout: {self.encoder.dropout.p!r}; "
                f"Decoder dropout: {self.decoder.dropout.p!r}.",
                category=RuntimeWarning,
                stacklevel=2,
            )
        if self.encoder.lstm.dropout != self.decoder.lstm.dropout:
            warnings.warn(
                "Mismatch in LSTM dropout probabilities between encoder and decoder. "
                "Saving the model with the encoder's LSTM dropout value for both. "
                f"Encoder LSTM dropout: {self.encoder.lstm.dropout!r}; "
                f"Decoder LSTM dropout: {self.decoder.lstm.dropout!r}.",
                category=RuntimeWarning,
                stacklevel=2,
            )

        serialized: dict[str, dict[str, Any]] = {
            "parameters": {
                "source_language": self.source_language,
                "target_language": self.target_language,
                "source_tokenizer": self.source_tokenizer,
                "target_tokenizer": self.target_tokenizer,
                "target_detokenizer": self.target_detokenizer,
                "source_embedding_size": self.encoder.embedding.embedding_dim,
                "target_embedding_size": self.decoder.embedding.embedding_dim,
                "hidden_size": self.encoder.lstm.hidden_size,
                "freeze_source_embeddings": not self.encoder.embedding.weight.requires_grad,
                "freeze_target_embeddings": not self.decoder.embedding.weight.requires_grad,
                "embedding_dropout": self.encoder.dropout.p,
                "dropout": self.encoder.lstm.dropout,
            },
            "state_dict": self.state_dict(),
        }

        torch.save(serialized, f)

    @classmethod
    def load(cls, f: str | Path | IO[bytes]) -> Self:
        serialized: dict[str, dict[str, Any]] = torch.load(f)

        parameters: dict[str, Any] = serialized["parameters"]
        translator: Self = cls(
            source_language=parameters["source_language"],
            target_language=parameters["target_language"],
            source_tokenizer=parameters["source_tokenizer"],
            target_tokenizer=parameters["target_tokenizer"],
            target_detokenizer=parameters["target_detokenizer"],
            source_embedding_size=parameters["source_embedding_size"],
            target_embedding_size=parameters["target_embedding_size"],
            hidden_size=parameters["hidden_size"],
            embedding_dropout=parameters["embedding_dropout"],
            dropout=parameters["dropout"],
        )

        translator.load_state_dict(serialized["state_dict"])

        if parameters["freeze_source_embeddings"]:
            translator.encoder.embedding.requires_grad_(requires_grad=False)
        if parameters["freeze_target_embeddings"]:
            translator.decoder.embedding.requires_grad_(requires_grad=False)

        translator.eval()
        return translator
