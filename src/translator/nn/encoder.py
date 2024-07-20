import torch
from gensim.models import KeyedVectors
from torch import Tensor
from torch.nn.utils import rnn

from translator.nn.module import Module
from translator.nn.utils import build_embedding_layer


class Encoder(Module):
    def __init__(
        self,
        vocabulary_size: int,
        embedding_size: int,
        hidden_size: int,
        *,
        pretrained_embeddings: KeyedVectors | Tensor | None = None,
        freeze_pretrained_embeddings: bool = True,
        padding_index: int | None = None,
    ) -> None:
        """Initializes the Encoder with an embedding layer and an LSTM.

        Args:
            vocabulary_size: The size of the source vocabulary.
            embedding_size: The size of the source embeddings.
            hidden_size: The size of the LSTM hidden state.
            pretrained_embeddings: Optional pretrained embeddings to initialize the embedding layer.
            freeze_pretrained_embeddings: If True, freezes the pretrained embeddings.
            padding_index: An optional index for the padding token.
                If specified, the padding entries do not contribute to the gradient.
        """
        super().__init__()

        self.padding_index = padding_index
        if padding_index is not None and padding_index < 0:
            msg: str = f"{type(self).__name__} does not support negative padding indices."
            raise ValueError(msg)

        self.embedding: torch.nn.Embedding = build_embedding_layer(
            vocabulary_size,
            embedding_size,
            pretrained_embeddings=pretrained_embeddings,
            freeze_pretrained_embeddings=freeze_pretrained_embeddings,
            padding_index=padding_index,
        )
        self.lstm: torch.nn.LSTM = torch.nn.LSTM(embedding_size, hidden_size)

    def _infer_sequence_length(self, sources: Tensor) -> Tensor:
        """Infers the sequence lengths of a sources tensor using the padding index.

        If the encoder does not define a padding index,
        the sequence lengths will uniformly be the size of the source's length dimension.

        Args:
            sources: A tensor of token indices. Shape: [batch_size, max(source_sequence_lengths)].

        Returns:
            The source's sequence lengths. Shape: [batch_size].
        """
        batch_size, max_sequence_length = sources.size()
        if self.padding_index is None:
            return torch.full(size=(batch_size,), fill_value=max_sequence_length)
        return max_sequence_length - sources.eq(self.padding_index).sum(dim=1)

    def forward(
        self,
        sources: Tensor,
        source_sequence_lengths: Tensor | None = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        """Forwards batches of (padded) sources as tensor of token indices through the encoder.

        Args:
            sources: A tensor of token indices. Shape: [batch_size, max(source_sequence_lengths)].
            source_sequence_lengths: An optional tensor of sequence lengths corresponding to the provided sources.
                If None, the sequence lengths will be inferred using the padding index. Shape: [batch_size].

        Returns:
            The encoder's LSTM hidden states (of shape: [batch_size x max(source_sequence_lengths) x hidden_size]) and
            the last (hidden, cell) state of the LSTM (each of shape: [1 x batch_size x hidden_size]).

        """
        if source_sequence_lengths is None:
            source_sequence_lengths = self._infer_sequence_length(sources)

        embedded: Tensor = self.embedding(sources)  # Shape: [batch_size, max(source_sequence_lengths), embedding_size]
        packed_input: rnn.PackedSequence = rnn.pack_padded_sequence(
            embedded,
            source_sequence_lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, last_hidden_and_cell = self.lstm(packed_input)
        hidden_states, _ = rnn.pad_packed_sequence(packed_output, batch_first=True)
        return hidden_states, last_hidden_and_cell
