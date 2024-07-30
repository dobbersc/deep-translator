import math
import warnings
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import IO, Any, Literal, Self, cast, overload

import more_itertools
import torch
from gensim.models import KeyedVectors
from sacrebleu.metrics.bleu import BLEU, BLEUScore
from torch import Tensor
from tqdm import tqdm

from translator.datasets import VectorizedDataPointBatch
from translator.language import Language
from translator.nn import DecoderLSTM, DotProductAttention, EncoderLSTM, Seq2Seq
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
        encoder_num_layers: int = 1,
        decoder_num_layers: int | None = None,
        bidirectional_encoder: bool = False,
        attention: bool = False,
        source_pretrained_embeddings: KeyedVectors | Tensor | None = None,
        target_pretrained_embeddings: KeyedVectors | Tensor | None = None,
        freeze_pretrained_embeddings: bool = True,
        embedding_dropout: float = 0,
        dropout: float = 0,
        propagate_hidden_and_cell_state: bool = True,
    ) -> None:
        decoder_hidden_size: int = 2 * hidden_size if bidirectional_encoder else hidden_size
        super().__init__(
            EncoderLSTM(
                vocabulary_size=source_language.vocabulary_size,
                embedding_size=source_embedding_size,
                hidden_size=hidden_size,
                num_layers=encoder_num_layers,
                bidirectional=bidirectional_encoder,
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
                hidden_size=decoder_hidden_size,
                num_layers=encoder_num_layers if decoder_num_layers is None else decoder_num_layers,
                bidirectional_encoder=bidirectional_encoder,
                # TODO: Support more attention variants. This may lead to problem regarding the serialization.
                attention=DotProductAttention(decoder_hidden_size) if attention else None,
                pretrained_embeddings=target_pretrained_embeddings,
                freeze_pretrained_embeddings=freeze_pretrained_embeddings,
                embedding_dropout=embedding_dropout,
                dropout=dropout,
                padding_index=target_language.padding_token_index,
                seperator_index=target_language.seperator_token_index,
                stop_index=target_language.stop_token_index,
            ),
            propagate_hidden_and_cell_state=propagate_hidden_and_cell_state,
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
        texts: list[str] | tuple[str, ...],
        method: Literal["greedy", "sampled", "beam-search"] = ...,
        max_length: int = ...,
        **kwargs: Any,
    ) -> list[str]:
        ...

    @torch.no_grad()
    def translate(
        self,
        texts: str | list[str] | tuple[str, ...],
        method: Literal["greedy", "sampled", "beam-search"] = "sampled",
        max_length: int = 512,
        **kwargs: Any,
    ) -> str | list[str]:
        self.eval()

        is_batched_input: bool = isinstance(texts, (list | tuple))
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

    def evaluate_bleu(
        self,
        sources: Sequence[str],
        targets: Sequence[Sequence[str]],
        *,
        lowercase: bool = False,
        max_ngram_order: int = 4,
        batch_size: int = 32,
    ) -> BLEUScore:
        """Calculates the BLEU score for the targets and the translation of the given sources.

        Args:
            sources: The source texts to be translated.
            targets: The valid target texts for source.
            max_ngram_order: Let BLEU consider all n-gram precisions, where 1 <= n <= max_ngram_order.
            lowercase: If True, the lowercased BLEU will be computed.
            batch_size: The batch size used for the translation model.

        Returns:
            An instance of the sacremoses `BLEUScore`.
        """
        source_batches: list[tuple[str, ...]] = list(more_itertools.batched(sources, n=batch_size))
        translated_sources: list[str] = [
            source
            for sources_batch in tqdm(source_batches, desc="Evaluating Model", unit="batch", leave=False)
            for source in self.translate(sources_batch)
        ]
        bleu: BLEU = BLEU(lowercase=lowercase, max_ngram_order=max_ngram_order)
        bleu_corpus_score: BLEUScore = bleu.corpus_score(hypotheses=translated_sources, references=targets)
        return bleu_corpus_score

    def save(self, f: str | Path | IO[bytes]) -> None:
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
                "encoder_num_layers": self.encoder.lstm.num_layers,
                "decoder_num_layers": self.decoder.lstm.num_layers,
                "bidirectional_encoder": self.encoder.lstm.bidirectional,
                "attention": self.decoder.attention is not None,
                "freeze_source_embeddings": not self.encoder.embedding.weight.requires_grad,
                "freeze_target_embeddings": not self.decoder.embedding.weight.requires_grad,
                "embedding_dropout": self.encoder.dropout.p,
                "dropout": self.encoder.lstm.dropout,
                "propagate_hidden_and_cell_state": self.propagate_hidden_and_cell_state,
            },
            "state_dict": self.state_dict(),
        }

        torch.save(serialized, f)

    @classmethod
    def load(cls, f: str | Path | IO[bytes], **kwargs: Any) -> Self:
        serialized: dict[str, dict[str, Any]] = torch.load(f, **kwargs)

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
            encoder_num_layers=parameters["encoder_num_layers"],
            decoder_num_layers=parameters["decoder_num_layers"],
            bidirectional_encoder=parameters["bidirectional_encoder"],
            attention=parameters["attention"],
            embedding_dropout=parameters["embedding_dropout"],
            dropout=parameters["dropout"],
            propagate_hidden_and_cell_state=parameters["propagate_hidden_and_cell_state"],
        )

        translator.load_state_dict(serialized["state_dict"])

        if parameters["freeze_source_embeddings"]:
            translator.encoder.embedding.requires_grad_(requires_grad=False)
        if parameters["freeze_target_embeddings"]:
            translator.decoder.embedding.requires_grad_(requires_grad=False)

        translator.eval()
        return translator


# TODO: Unify class interface of Translator and PivotTranslator.
# TODO: The PivotTranslator may also accept other PivotTranslators ans source and target translators.


class PivotTranslator:
    def __init__(self, source2pivot: Translator, pivot2target: Translator) -> None:
        if source2pivot.target_language.name != pivot2target.source_language.name:
            msg: str = (
                "The target language of the source2pivot translator must match "
                "the source language of the pivot2target translator."
            )
            raise ValueError(msg)

        self.source2pivot = source2pivot
        self.pivot2target = pivot2target

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
        texts: list[str] | tuple[str, ...],
        method: Literal["greedy", "sampled", "beam-search"] = ...,
        max_length: int = ...,
        **kwargs: Any,
    ) -> list[str]:
        ...

    def translate(
        self,
        texts: str | list[str] | tuple[str, ...],
        method: Literal["greedy", "sampled", "beam-search"] = "sampled",
        max_length: int = 512,
        **kwargs: Any,
    ) -> str | list[str]:
        pivots = self.source2pivot.translate(texts, method=method, max_length=max_length, **kwargs)
        return self.pivot2target.translate(pivots, method=method, max_length=max_length, **kwargs)

    @property
    def source_language(self) -> str:
        return self.source2pivot.source_language.name

    @property
    def pivot_language(self) -> str:
        return self.source2pivot.target_language.name

    @property
    def target_language(self) -> str:
        return self.pivot2target.target_language.name

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}<"
            f"source_language={self.source_language!r}, "
            f"pivot_language={self.pivot_language!r}, "
            f"target_language={self.target_language!r}>"
        )
