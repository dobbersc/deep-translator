from collections.abc import Callable, Iterable, Sequence

import math
import torch
from gensim.models import KeyedVectors
from torch import Tensor
from tqdm import tqdm
from translator.datasets import VectorizedDataPointBatch
from translator.language import Language
from translator.nn import DecoderLSTM, EncoderLSTM, Seq2Seq
from translator.tokenizers import Tokenizer, Detokenizer


class Translator(Seq2Seq):
    def __init__(
        self,
        *,
        source_language: Language,
        target_language: Language,
        source_tokenizer: Tokenizer,
        target_tokenizer: Tokenizer,
        target_detokenizer: Detokenizer = lambda x: " ".join(x),
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

    def translate(self, text: str) -> str:
        raise NotImplementedError

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

    def evaluate_bleu(self, sources: Sequence[str], targets: Sequence[str], batch_size: int = 32) -> float:
        raise NotImplementedError
