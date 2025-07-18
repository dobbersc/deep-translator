import torch
from gensim.models import KeyedVectors
from torch import Tensor
from torch.nn.utils import rnn

from translator.nn.module import Module
from translator.nn.utils import build_embedding_layer


class EncoderLSTM(Module):
    def __init__(
        self,
        vocabulary_size: int,
        embedding_size: int,
        hidden_size: int,
        *,
        num_layers: int = 1,
        bidirectional: bool = False,
        pretrained_embeddings: KeyedVectors | Tensor | None = None,
        freeze_pretrained_embeddings: bool = True,
        embedding_dropout: float = 0,
        dropout: float = 0,
        padding_index: int | None = None,
        start_index: int = 1,
    ) -> None:
        """Initializes the Encoder with an embedding layer and an LSTM.

        Args:
            vocabulary_size: The size of the source vocabulary.
            embedding_size: The size of the source embeddings.
            hidden_size: The size of the LSTM hidden state.
            num_layers: The number of LSTM's recurrent layers.
            bidirectional: If true, a bidirectional LSTM will be used.
            pretrained_embeddings: Optional pretrained embeddings to initialize the embedding layer.
            freeze_pretrained_embeddings: If True, freezes the pretrained embeddings.
            embedding_dropout: If non-zero, introduces a dropout layer on the outputs of the embedding layer,
                with dropout probability equal to `embedding_dropout`.
            dropout: If non-zero, introduces dropout to the LSTM, with dropout probability equal to `dropout`.
            padding_index: An optional index for the padding token.
                If specified, the padding entries do not contribute to the gradient.
            start_index: The index of the special token that marks the source's start.
        """
        super().__init__()

        self.embedding: torch.nn.Embedding = build_embedding_layer(
            vocabulary_size,
            embedding_size,
            pretrained_embeddings=pretrained_embeddings,
            freeze_pretrained_embeddings=freeze_pretrained_embeddings,
            padding_index=padding_index,
        )
        self.dropout: torch.nn.Dropout = torch.nn.Dropout(embedding_dropout)
        self.lstm: torch.nn.LSTM = torch.nn.LSTM(
            embedding_size,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=True,
        )

        self.padding_index = padding_index
        self.start_index = start_index

    def make_input_sources(self, sources: Tensor) -> Tensor:
        """Encodes the sources as input for the encoder by prepending each source with the start special token.

        Args:
            sources: A tensor of token indices. Shape: [batch_size, max(source_sequence_lengths)].

        Returns:
            The encoded sources. Shape: [batch_size, max(source_sequence_lengths) + 1].
        """
        start_tokens: Tensor = torch.full(
            size=(sources.size(dim=0), 1),
            fill_value=self.start_index,
            dtype=sources.dtype,
            device=sources.device,
        )
        return torch.cat((start_tokens, sources), dim=1)

    def make_padding_mask(self, sources: Tensor) -> Tensor:
        """Creates a padding mask for a batch of source sequences.

        The padding mask accounts for the start special token by prepending `False` to each source sequence's mask.

        Args:
            sources: A tensor of token indices (excluding the start special token).
                Shape: [batch_size, max(source_sequence_lengths)].

        Returns:
            A boolean tensor indicating the positions of padding tokens in the sources.
            Shape: [batch_size, max(source_sequence_lengths) + 1].
        """
        batch_size, max_sequence_length = sources.size()
        if self.padding_index is None:
            return torch.full(
                (batch_size, max_sequence_length + 1),
                fill_value=False,
                dtype=torch.bool,
                device=sources.device,
            )

        sources_mask: Tensor = sources.eq(self.padding_index)
        start_token_mask: Tensor = torch.full(
            size=(batch_size, 1),
            fill_value=False,
            dtype=torch.bool,
            device=sources.device,
        )
        return torch.cat((start_token_mask, sources_mask), dim=1)

    def infer_sequence_lengths(self, sources: Tensor) -> Tensor:
        """Infers the sequence lengths of a sources tensor using the padding index.

        If the encoder does not define a padding index,
        the sequence lengths will uniformly be the size of the source's length dimension.

        Args:
            sources: A tensor of token indices. Shape: [batch_size, max(source_sequence_lengths)].

        Returns:
            The source's sequence lengths as tensor on the CPU. Shape: [batch_size].
        """
        batch_size, max_sequence_length = sources.size()
        if self.padding_index is None:
            return torch.full(size=(batch_size,), fill_value=max_sequence_length, dtype=torch.long, device="cpu")
        sequence_lengths: Tensor = max_sequence_length - sources.eq(self.padding_index).sum(dim=1).long().cpu()
        return sequence_lengths

    def forward(
        self,
        sources: Tensor,
        source_sequence_lengths: Tensor | None = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        """Forwards batches of (padded) sources as tensor of token indices through the encoder.

        Args:
            sources: A tensor of token indices. Shape: [batch_size, max(source_sequence_lengths)].
            source_sequence_lengths: An optional tensor of sequence lengths corresponding to the provided sources.
                If None, the sequence lengths will be inferred using the padding index.
                Must be a CPU tensor of type `torch.long`. Shape: [batch_size].

        Returns:
            The encoder's LSTM hidden states (of shape: [batch_size, max(source_sequence_lengths), D * hidden_size]) and
            the last (hidden, cell) state of the LSTM (each of shape: [D * num_layers, batch_size, hidden_size]).
            D := 2 if bidirectional; otherwise 1.
        """
        sources = self.make_input_sources(sources)
        source_sequence_lengths = (
            self.infer_sequence_lengths(sources)
            if source_sequence_lengths is None
            else source_sequence_lengths + 1  # +1 for the start special token
        )

        # Shape: [batch_size, max(source_sequence_lengths), embedding_size].
        embedded: Tensor = self.embedding(sources)
        embedded = self.dropout(embedded)

        packed_input: rnn.PackedSequence = rnn.pack_padded_sequence(
            embedded,
            source_sequence_lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, last_hidden_and_cell = self.lstm(packed_input)
        hidden_states, _ = rnn.pad_packed_sequence(packed_output, batch_first=True)
        return hidden_states, last_hidden_and_cell
