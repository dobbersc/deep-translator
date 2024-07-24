import random
from typing import Any, Literal

import torch
from gensim.models import KeyedVectors
from torch import Tensor

from translator.nn.module import Module
from translator.nn.utils import build_embedding_layer


class DecoderLSTM(Module):
    def __init__(
        self,
        vocabulary_size: int,
        embedding_size: int,
        hidden_size: int,
        *,
        pretrained_embeddings: KeyedVectors | Tensor | None = None,
        freeze_pretrained_embeddings: bool = True,
        embedding_dropout: float = 0,
        dropout: float = 0,
        padding_index: int | None = None,
        seperator_index: int = 2,
        stop_index: int = 3,
    ) -> None:
        """Initializes the Decoder with an embedding layer and an LSTM.

        Args:
            vocabulary_size: The size of the target vocabulary.
            embedding_size: The size of the target embeddings.
            hidden_size: The size of the LSTM hidden state.
            pretrained_embeddings: Optional pretrained embeddings to initialize the embedding layer.
            freeze_pretrained_embeddings: If True, freezes the pretrained embeddings.
            embedding_dropout: If non-zero, introduces a dropout layer on the outputs of the embedding layer,
                with dropout probability equal to `embedding_dropout`.
            dropout: If non-zero, introduces dropout to the LSTM, with dropout probability equal to `dropout`.
            padding_index: An optional index for the padding token.
                If specified, the padding entries do not contribute to the gradient.
            seperator_index: The index of the special token that marks the target's start.
            stop_index: The index of the special token that marks the target's end.
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
        self.lstm: torch.nn.LSTM = torch.nn.LSTM(embedding_size, hidden_size, dropout=dropout, batch_first=True)
        self.hidden2vocab: torch.nn.Linear = torch.nn.Linear(hidden_size, vocabulary_size)
        self.log_softmax: torch.nn.LogSoftmax = torch.nn.LogSoftmax(dim=-1)

        self.padding_index = padding_index
        self.seperator_index = seperator_index
        self.stop_index = stop_index

    def make_input_sequences(self, targets: Tensor) -> Tensor:
        """Encodes the targets as input for the decoder by prepending each with the seperator special token.

        Args:
            targets: A tensor of token indices. Shape: [batch_size, max(target_sequence_lengths)].

        Returns:
            The encoded targets. Shape: [batch_size, max(target_sequence_lengths) + 1].
        """
        seperator_tokens: Tensor = torch.full(
            size=(targets.size(dim=0), 1),
            fill_value=self.seperator_index,
            dtype=targets.dtype,
            device=targets.device,
        )
        return torch.cat((seperator_tokens, targets), dim=1)

    def make_ground_truth_sequences(self, targets: Tensor) -> Tensor:
        """Encodes the targets as ground truth output of the decoder by appending each with the stop special token.

        Args:
            targets: A tensor of token indices. Shape: [batch_size, max(target_sequence_lengths)].

        Returns:
            The encoded targets. Shape: [batch_size, max(target_sequence_lengths) + 1].
        """
        stop_tokens: Tensor = torch.full(
            size=(targets.size(dim=0), 1),
            fill_value=self.stop_index,
            dtype=targets.dtype,
            device=targets.device,
        )
        return torch.cat((targets, stop_tokens), dim=1)

    def step(
        self,
        decoder_input: Tensor,
        hidden_and_cell: tuple[Tensor, Tensor] | None = None,
        encoder_hidden_states: Tensor | None = None,  # noqa: ARG002
    ) -> tuple[Tensor, tuple[Tensor, Tensor], Tensor | None]:
        """Performs a single decoding step through the decoder.

        Args:
            decoder_input: The input token indices for the decoder at the current time step. Shape: [batch_size].
            hidden_and_cell: The hidden and cell state for the LSTM (each of shape [1, batch_size, hidden_size]).
            encoder_hidden_states: TODO: Placeholder for later attention

        Returns:
            - log_probabilities: The predicted log softmax probabilities for the next token in the target vocabulary.
                Shape: [batch_size, target_vocabulary_size].
            - last_hidden_and_cell: The updated (hidden, cell) state of the LSTM after processing the input
                (each of shape: [1, batch_size, hidden_size]).
            - None: TODO: Placeholder for later attention
        """
        # Reshape the decoder inputs to tensors of sequence length 1 for the LSTM. Shape: [batch_size, 1].
        decoder_input = decoder_input.unsqueeze(dim=1)

        # Shape: [batch_size, 1, embedding_size].
        embedded: Tensor = self.embedding(decoder_input)
        embedded = self.dropout(embedded)

        decoder_hidden_states, last_hidden_and_cell = self.lstm(embedded, hidden_and_cell)

        # Transform the batch's hidden states to the vocabulary space and
        # obtain the log softmax probabilities for each token. Shape: [batch_size, 1, target_vocabulary_size].
        logits: Tensor = self.hidden2vocab(decoder_hidden_states)
        log_probabilities: Tensor = self.log_softmax(logits)

        return log_probabilities.squeeze(dim=1), last_hidden_and_cell, None

    def forward(
        self,
        targets: Tensor,
        encoder_hidden_states: Tensor | None = None,
        encoder_hidden_and_cell: tuple[Tensor, Tensor] | None = None,
        teacher_forcing_ratio: float = 0.5,
    ) -> Tensor:
        """Forwards batches of (padded) targets as tensor of token indices through the decoder.

        TODO: Decide if the None values make sense

        Args:
            targets: A tensor of token indices. Shape: [batch_size, max(target_sequence_lengths)].
            encoder_hidden_states: TODO: Placeholder for later attention
            encoder_hidden_and_cell: An optional initial hidden and cell state from the encoder for the LSTM
                (each of shape [1, batch_size, hidden_size]).
            teacher_forcing_ratio: The probability that teacher forcing will be used.
                For each decoding token, a random number is drawn uniformly from the interval [0, 1).
                If the random number is below the specified value, teacher forcing will be applied (default is 0.5).

        Returns:
            The predicted log softmax probabilities in the target vocabulary for each token in the target sequence.
            Shape: [batch_size, max(target_sequence_lengths), target_vocabulary_size].
        """
        targets = self.make_input_sequences(targets)

        # Create a tensor that contains the log probabilities for each predicted token.
        # For now, the batch_size is in the second dimension!
        (batch_size, target_sequence_length), target_vocabulary_size = targets.shape, self.embedding.num_embeddings
        prediction_log_probabilities: Tensor = torch.empty(
            (target_sequence_length, batch_size, target_vocabulary_size),
            dtype=targets.dtype,
            device=self.device,
        )

        decoder_output: Tensor | None = None  # Shape: [batch_size, target_vocabulary_size]

        # If provided, the decoder's initial (hidden, cell) state is the encoder's last (hidden, cell) state.
        decoder_hidden_and_cell: tuple[Tensor, Tensor] | None = encoder_hidden_and_cell

        # Iterate batch-wise through every token in the decoder input sequences and
        # populate the prediction log probabilities tensor. Shape: [batch_size].
        for sequence_index, decoder_input in enumerate(targets.transpose(0, 1)):
            use_teacher_forcing: bool = decoder_output is None or random.random() < teacher_forcing_ratio

            # If teacher forcing is enabled for this iteration, use the true target token
            # as the input for the next decoder step. Otherwise, use the predicted token from the previous step.
            # TODO: Check if detaching the decoder outputs is useful:
            #   https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
            effective_decoder_input: Tensor = (
                decoder_input if use_teacher_forcing else decoder_output.argmax(dim=1).detach()  # type: ignore[union-attr]
            )

            decoder_output, decoder_hidden_and_cell, _ = self.step(
                effective_decoder_input,
                hidden_and_cell=decoder_hidden_and_cell,
                encoder_hidden_states=encoder_hidden_states,
            )
            prediction_log_probabilities[sequence_index] = decoder_output

        # Transpose the prediction log probabilities back to a batch first representation.
        # Shape: [batch_size, max(target_sequence_lengths), target_vocabulary_size].
        return prediction_log_probabilities.transpose(0, 1)

    def decode_greedy(
        self,
        encoder_hidden_states: Tensor,
        encoder_hidden_and_cell: tuple[Tensor, Tensor],
        max_length: int = 512,
    ) -> Tensor:
        raise NotImplementedError

    def decode_sampled(
        self,
        encoder_hidden_states: Tensor,
        encoder_hidden_and_cell: tuple[Tensor, Tensor],
        temperature: float = 0.2,
        max_length: int = 512,
    ) -> Tensor:
        raise NotImplementedError

    def decode_beam_search(
        self,
        encoder_hidden_states: Tensor,
        encoder_hidden_and_cell: tuple[Tensor, Tensor],
        max_length: int = 512,
    ) -> Tensor:
        raise NotImplementedError

    def decode(
        self,
        encoder_hidden_states: Tensor,
        encoder_hidden_and_cell: tuple[Tensor, Tensor],
        method: Literal["greedy", "sampled", "beam-search"] = "sampled",
        **kwargs: Any,
    ) -> Tensor:
        if method == "greedy":
            return self.decode_greedy(encoder_hidden_states, encoder_hidden_and_cell, **kwargs)
        if method == "sampled":
            return self.decode_sampled(encoder_hidden_states, encoder_hidden_and_cell, **kwargs)
        if method == "beam-search":
            return self.decode_beam_search(encoder_hidden_states, encoder_hidden_and_cell, **kwargs)
        raise ValueError
