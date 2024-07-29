import random
from collections.abc import Callable
from typing import Any, Literal

import torch
from gensim.models import KeyedVectors
from torch import Tensor

from translator.nn.attention import Attention
from translator.nn.module import Module
from translator.nn.utils import build_embedding_layer


class DecoderLSTM(Module):
    """Implementation of a Luong Decoder.

    [Effective Approaches to Attention-based Neural Machine Translation](https://aclanthology.org/D15-1166)
    (Luong et al., EMNLP 2015)
    """

    def __init__(
        self,
        vocabulary_size: int,
        embedding_size: int,
        hidden_size: int,
        *,
        num_layers: int = 1,
        bidirectional_encoder: bool = False,
        attention: Attention | None = None,
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
            num_layers: The number of LSTM's recurrent layers.
            bidirectional_encoder: Flag to indicate whether the connected encoder is bidirectional.
                See the decoder's forward method for more information about the flag's influences.
            attention: An optional attention module.
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
        self.lstm: torch.nn.LSTM = torch.nn.LSTM(
            embedding_size,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.attention = attention
        self.hidden2vocab: torch.nn.Linear = torch.nn.Linear(hidden_size, vocabulary_size)
        self.log_softmax: torch.nn.LogSoftmax = torch.nn.LogSoftmax(dim=-1)

        self.bidirectional_encoder = bidirectional_encoder

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

    @staticmethod
    def _transfer_directions_to_hidden_size(hidden_or_cell: Tensor) -> Tensor:
        """Concatenates the forward and backward directions of the hidden and cell state in the `hidden_size` dimension.

        Args:
            hidden_or_cell: A hidden and cell state from a bidirectional encoder
                (each of shape [2 * encoder_num_layers, batch_size, encoder_hidden_size].

        Returns:
            The transformed hidden and cell state. Shape: [encoder_num_layers, batch_size, 2 * encoder_hidden_size].
        """
        directions_times_num_layers = hidden_or_cell.size(dim=0)
        return torch.cat(
            (
                hidden_or_cell[0:directions_times_num_layers:2],
                hidden_or_cell[1:directions_times_num_layers:2],
            ),
            dim=2,
        )

    def _init_decoder_hidden_and_cell(
        self,
        encoder_hidden_and_cell: tuple[Tensor, Tensor] | None,
    ) -> tuple[Tensor, Tensor] | None:
        """Initializes the decoder's initial hidden and cell state from the encoder's last hidden and cell state.

        If the encoder is bidirectional, the forward and backward directions of its
        hidden and cell state get concatenated in the `hidden_size` dimension.
        From the resulting states, the top `decoder_num_layers` states will be used as the decoder's initial state.

        # TODO: Other ideas for the case where decoder_num_layers <= encoder_num_layers:
            - Average the encoder's hidden and cell state and use the same hidden and cell state for all decoder layers.
            - Employ a linear projection layer and use the same hidden and cell state for all decoder layers.

        # TODO: We could also allow decoder_num_layers > encoder_num_layers,
            by initializing the remaining hidden and cell states with zeros.

        Args:
            encoder_hidden_and_cell: A hidden and cell state from the encoder
                (each of shape [D * encoder_num_layers, batch_size, encoder_hidden_size].

        Returns:
            The decoder's initial hidden and cell state
            (each of shape: [decoder_num_layers, batch_size, decoder_hidden_size],
            where decoder_hidden_size := D * encoder_hidden_size and decoder_num_layers <= encoder_num_layers).
        """
        if encoder_hidden_and_cell is None:
            return None

        hidden, cell = encoder_hidden_and_cell
        if self.bidirectional_encoder:
            hidden = self._transfer_directions_to_hidden_size(hidden)
            cell = self._transfer_directions_to_hidden_size(cell)

        hidden = hidden[-self.lstm.num_layers :]
        cell = cell[-self.lstm.num_layers :]

        return hidden, cell

    def step(
        self,
        decoder_input: Tensor,
        hidden_and_cell: tuple[Tensor, Tensor] | None = None,
        encoder_hidden_states: Tensor | None = None,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor], tuple[Tensor, Tensor] | None]:
        """Performs a single decoding step through the decoder.

        Args:
            decoder_input: The input token indices for the decoder at the current time step. Shape: [batch_size].
            hidden_and_cell: A optional hidden and cell state for the decoder's LSTM
                (each of shape [decoder_num_layers, batch_size, decoder_hidden_size]).
            encoder_hidden_states: Optional encoder's LSTM hidden states used for the attention computation.
                These are only required if attention is enabled.
                Shape: [batch_size, max(source_sequence_lengths), D * encoder_hidden_size],
                where decoder_hidden_size := D * encoder_hidden_size.
            attention_mask: An optional attention mask to exclude source tokens from the attention computation,
                e.g. padding tokens. The mask will only be effective if attention is enabled.
                Shape: [batch_size, max(source_sequence_lengths)].

        Returns:
            - log_probabilities: The predicted log softmax probabilities for the next token in the target vocabulary.
              Shape: [batch_size, target_vocabulary_size].
            - last_hidden_and_cell: The updated (hidden, cell) state of the decoder's LSTM after processing the input
              (each of shape: [decoder_num_layers, batch_size, decoder_hidden_size]).
            - attention_scores_and_distribution: A tuple of the attention scores and attention distribution
              if attention is enabled. Otherwise, this value will be `None` for compatibility.
              Each of shape [batch_size, max(source_sequence_lengths)].
        """
        if self.attention is not None and encoder_hidden_states is None:
            msg: str = "The 'encoder_hidden_states' parameter must be provided when attention is enabled."
            raise ValueError(msg)

        # Reshape the decoder inputs to tensors of sequence length 1 for the LSTM. Shape: [batch_size, 1].
        decoder_input = decoder_input.unsqueeze(dim=1)

        # Shape: [batch_size, 1, embedding_size].
        embedded: Tensor = self.embedding(decoder_input)
        embedded = self.dropout(embedded)

        output, last_hidden_and_cell = self.lstm(embedded, hidden_and_cell)

        # Compute attention.
        attention_scores_and_distribution: tuple[Tensor, Tensor] | None = None
        if self.attention:
            output, attention_scores_and_distribution = self.attention(
                output=output,
                context=encoder_hidden_states,
                mask=None if attention_mask is None else attention_mask.unsqueeze(dim=1),
            )
            assert attention_scores_and_distribution is not None
            attention_scores_and_distribution = (
                attention_scores_and_distribution[0].squeeze(dim=1),
                attention_scores_and_distribution[1].squeeze(dim=1),
            )

        # Transform the batch's hidden states to the vocabulary space and
        # obtain the log softmax probabilities for each token. Shape: [batch_size, 1, target_vocabulary_size].
        logits: Tensor = self.hidden2vocab(output)
        log_probabilities: Tensor = self.log_softmax(logits)

        return log_probabilities.squeeze(dim=1), last_hidden_and_cell, attention_scores_and_distribution

    def forward(
        self,
        targets: Tensor,
        encoder_hidden_states: Tensor | None = None,
        encoder_hidden_and_cell: tuple[Tensor, Tensor] | None = None,
        attention_mask: Tensor | None = None,
        teacher_forcing_ratio: float = 0.5,
    ) -> tuple[Tensor, tuple[Tensor, Tensor] | None]:
        """Forwards batches of (padded) targets as tensor of token indices through the decoder.

        For the following argument descriptions, we define D := 2 if bidirectional_encoder; otherwise 1.

        Args:
            targets: A tensor of token indices. Shape: [batch_size, max(target_sequence_lengths)].
            encoder_hidden_states: Optional encoder's LSTM hidden states used for the attention computation.
                These are only required if attention is enabled.
                Shape: [batch_size, max(source_sequence_lengths), D * encoder_hidden_size],
                where decoder_hidden_size := D * encoder_hidden_size.
            encoder_hidden_and_cell: An optional initial hidden and cell state from the encoder for the LSTM
                (each of shape [D * encoder_num_layers, batch_size, encoder_hidden_size],
                where decoder_hidden_size := D * encoder_hidden_size and decoder_num_layers <= encoder_num_layers).
                If the encoder is bidirectional, the forward and backward directions of its
                hidden and cell state get concatenated in the `hidden_size` dimension. From the resulting states,
                the top `decoder_num_layers` states will be used as the decoder's initial state.
            attention_mask: An optional attention mask to exclude source tokens from the attention computation,
                e.g. padding tokens. The mask will only be effective if attention is enabled.
                Shape: [batch_size, max(source_sequence_lengths)].
            teacher_forcing_ratio: The probability that teacher forcing will be used.
                For each decoding token, a random number is drawn uniformly from the interval [0, 1).
                If the random number is below the specified value, teacher forcing will be applied (default is 0.5).

        Returns:
            - prediction_log_probabilities: The predicted log softmax probabilities in the target vocabulary for each
              token in the target sequence. Shape: [batch_size, max(target_sequence_lengths), target_vocabulary_size].
            - attention_scores_and_distributions: A tuple of the attention scores and attention distributions
              if attention is enabled. Otherwise, this value will be `None` for compatibility.
              Each of shape [batch_size, max(target_sequence_lengths), max(source_sequence_lengths)].
        """
        msg: str
        if self.attention is None and encoder_hidden_and_cell is None:
            msg = "The 'encoder_hidden_and_cell' parameter must be provided when attention disabled."
            raise ValueError(msg)

        if self.attention is not None and encoder_hidden_states is None:
            msg = "The 'encoder_hidden_states' parameter must be provided when attention is enabled."
            raise ValueError(msg)

        targets = self.make_input_sequences(targets)

        # Create a tensor that contains the log probabilities for each predicted token.
        # For now, the batch_size is in the second dimension!
        (batch_size, target_sequence_length), target_vocabulary_size = targets.shape, self.embedding.num_embeddings
        prediction_log_probabilities: Tensor = torch.empty(
            (target_sequence_length, batch_size, target_vocabulary_size),
            dtype=torch.float,
            device=self.device,
        )

        # Create tensors that contains the attention scores and distributions for each predicted token.
        # For now, the batch_size is in the second dimension!
        attention_scores: Tensor | None = None
        attention_distributions: Tensor | None = None
        if self.attention is not None:
            assert encoder_hidden_states is not None

            source_sequence_length: int = encoder_hidden_states.size(dim=1)
            attention_scores = torch.empty(
                (target_sequence_length, batch_size, source_sequence_length),
                dtype=torch.float,
                device=self.device,
            )
            attention_distributions = torch.empty(
                (target_sequence_length, batch_size, source_sequence_length),
                dtype=torch.float,
                device=self.device,
            )

        # If provided, the decoder's initial (hidden, cell) state is the encoder's last (hidden, cell) state.
        decoder_hidden_and_cell: tuple[Tensor, Tensor] | None = self._init_decoder_hidden_and_cell(
            encoder_hidden_and_cell,
        )
        decoder_output: Tensor | None = None  # Shape: [batch_size, target_vocabulary_size]

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

            decoder_output, decoder_hidden_and_cell, attention_scores_and_distribution = self.step(
                effective_decoder_input,
                hidden_and_cell=decoder_hidden_and_cell,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
            )
            prediction_log_probabilities[sequence_index] = decoder_output
            if attention_scores_and_distribution is not None:
                assert attention_scores is not None
                assert attention_distributions is not None
                attention_scores[sequence_index] = attention_scores_and_distribution[0]
                attention_distributions[sequence_index] = attention_scores_and_distribution[1]

        # Transpose the prediction log probabilities,
        # attention scores and distributions back to a batch first representation.
        attention_scores_and_distributions: tuple[Tensor, Tensor] | None = None
        if attention_scores is not None and attention_distributions is not None:
            attention_scores_and_distributions = (
                attention_scores.transpose(0, 1),
                attention_distributions.transpose(0, 1),
            )
        return prediction_log_probabilities.transpose(0, 1), attention_scores_and_distributions

    def _decode_sequential(
        self,
        encoder_hidden_states: Tensor | None,
        encoder_hidden_and_cell: tuple[Tensor, Tensor] | None,
        attention_mask: Tensor | None,
        aggregation_function: Callable[[Tensor], Tensor],
        max_length: int,
    ) -> Tensor:
        batch_size: int
        if encoder_hidden_states is not None:
            batch_size = encoder_hidden_states.size(dim=0)
        elif encoder_hidden_and_cell is not None:
            batch_size = encoder_hidden_and_cell[0].size(dim=1)
        else:
            msg: str = (
                "Decoding requires either 'encoder_hidden_states' or "
                "'encoder_hidden_and_cell' or both, but none were provided."
            )
            raise ValueError(msg)

        # The initial token is always the seperator token.
        current_tokens: Tensor = torch.full(
            size=(batch_size,),
            fill_value=self.seperator_index,
            dtype=torch.long,
            device=self.device,
        )

        # Use a boolean list as an indicator when all predicted sequences in the batch have reached the stop token.
        stopped: list[bool] = [False] * batch_size
        predicted_token_indices: list[Tensor] = []

        decoder_hidden_and_cell: tuple[Tensor, Tensor] | None = self._init_decoder_hidden_and_cell(
            encoder_hidden_and_cell,
        )
        for _ in range(max_length):
            predicted_log_probabilities, decoder_hidden_and_cell, _ = self.step(
                current_tokens,
                hidden_and_cell=decoder_hidden_and_cell,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
            )

            # Use the aggregation function to obtain the predicted batch of tokens from the log probabilities and
            # re-inject the predicted tokens into the decoder for the next iteration.
            current_tokens = aggregation_function(predicted_log_probabilities)
            predicted_token_indices.append(current_tokens)

            # Update which sentences in the batch have reached the stop symbol
            for batch_index, token_index in enumerate(current_tokens):
                if token_index == self.stop_index:
                    stopped[batch_index] = True

            # Exit the loop if all sequences have reached the stop token.
            if all(stopped):
                break

        return torch.stack(predicted_token_indices, dim=1)  # Shape: [batch_size, sequence_length]

    def decode_greedy(
        self,
        encoder_hidden_states: Tensor | None = None,
        encoder_hidden_and_cell: tuple[Tensor, Tensor] | None = None,
        attention_mask: Tensor | None = None,
        max_length: int = 512,
    ) -> Tensor:
        def aggregation_function(log_probabilities: Tensor) -> Tensor:
            return log_probabilities.argmax(dim=1)

        return self._decode_sequential(
            encoder_hidden_states,
            encoder_hidden_and_cell,
            attention_mask=attention_mask,
            aggregation_function=aggregation_function,
            max_length=max_length,
        )

    def decode_sampled(
        self,
        encoder_hidden_states: Tensor | None = None,
        encoder_hidden_and_cell: tuple[Tensor, Tensor] | None = None,
        attention_mask: Tensor | None = None,
        temperature: float = 0.8,
        max_length: int = 512,
    ) -> Tensor:
        def aggregation_function(log_probabilities: Tensor) -> Tensor:
            token_weights: Tensor = log_probabilities.div(temperature).exp()
            return torch.multinomial(token_weights, num_samples=1).squeeze(dim=1)

        return self._decode_sequential(
            encoder_hidden_states,
            encoder_hidden_and_cell,
            attention_mask=attention_mask,
            aggregation_function=aggregation_function,
            max_length=max_length,
        )

    def decode_beam_search(
        self,
        encoder_hidden_states: Tensor | None,
        encoder_hidden_and_cell: tuple[Tensor, Tensor] | None,
        attention_mask: Tensor | None = None,
        max_length: int = 512,
    ) -> Tensor:
        raise NotImplementedError

    def decode(
        self,
        encoder_hidden_states: Tensor | None = None,
        encoder_hidden_and_cell: tuple[Tensor, Tensor] | None = None,
        attention_mask: Tensor | None = None,
        method: Literal["greedy", "sampled", "beam-search"] = "sampled",
        **kwargs: Any,
    ) -> Tensor:
        standard_arguments: dict[str, Any] = {
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_hidden_and_cell": encoder_hidden_and_cell,
            "attention_mask": attention_mask,
        }
        if method == "greedy":
            return self.decode_greedy(**standard_arguments, **kwargs)
        if method == "sampled":
            return self.decode_sampled(**standard_arguments, **kwargs)
        if method == "beam-search":
            return self.decode_beam_search(**standard_arguments, **kwargs)

        msg: str = f"Invalid decoding method {method!r}. Choose from 'greedy', 'sampled' or 'beam-search'."  # type: ignore[unreachable]
        raise ValueError(msg)
