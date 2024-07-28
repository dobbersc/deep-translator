from abc import ABC, abstractmethod

import torch
from torch import Tensor

from translator.nn.module import Module


class Attention(Module, ABC):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()

        self.softmax = torch.nn.Softmax(dim=-1)
        self.combined2hidden: torch.nn.Sequential = torch.nn.Sequential(
            torch.nn.Linear(2 * hidden_size, hidden_size),
            torch.nn.ReLU(),
        )

    @abstractmethod
    def score(self, output: Tensor, context: Tensor) -> Tensor:
        """Computes the attention scores between the output and context tensor.

        Args:
            output: A tensor containing output features.
                For example, the output tensor from a decoder's LSTM for a single timestep.
                Shape: [batch_size, output_sequence_length, hidden_size].
            context: A tensor containing context features.
                For example, the hidden states from an encoder's LSTM for a source sequence.
                Shape: [batch_size, context_sequence_length, hidden_size].

        Returns:
            The attention scores. Shape: [batch_size, output_sequence_length, context_sequence_length].
        """

    def forward(
        self,
        output: Tensor,
        context: Tensor,
        mask: Tensor | None = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        """Computes the combined output and attention features and projects them to the hidden space.

        Args:
            output: A tensor containing output features.
                For example, the output tensor from a decoder's LSTM for a single timestep.
                Shape: [batch_size, output_sequence_length, hidden_size].
            context: A tensor containing context features.
                For example, the hidden states from an encoder's LSTM for a source sequence.
                Shape: [batch_size, context_sequence_length, hidden_size].
            mask: An optional tensor mask applied to the attention scores. Masked elements are excluded from the
                attention distribution, i.e. they have a probability of zero. For example, this can be used to exclude
                padding indices in the context tensor from the attention computation.
                Shape: [batch_size, output_sequence_length, context_sequence_length].

        Returns:
            - The combined output and attention features projected to the hidden space.
              Shape: [batch_size, output_sequence_length, hidden_size].
            - A tuple of the attention scores and attention distribution
              (each of shape [batch_size, output_sequence_length, context_sequence_length]).
        """
        # Shape: [batch_size, output_sequence_length, context_sequence_length].
        attention_scores: Tensor = self.score(output, context)
        if mask is not None:
            # We set the masked elements to -inf before applying the softmax
            # to exclude them from the attention distribution. Thus, we ensure that the masked elements have
            # an attention probability of zero while maintaining a valid probability distribution.
            # Setting them to zero after softmax would disrupt the distribution, causing the rows to not sum to one.
            attention_scores = attention_scores.masked_fill(mask, float("-inf"))

        attention_distribution: Tensor = self.softmax(attention_scores)

        # Compute the weighted sum of the context features by the attention distribution.
        #  | [batch_size, output_sequence_length,  context_sequence_length]
        #  @ [batch_size, context_sequence_length, hidden_size]
        # -> [batch_size, output_sequence_length,  hidden_size]
        attention_output: torch.Tensor = torch.bmm(attention_distribution, context)

        combined: torch.Tensor = torch.cat((attention_output, output), dim=2)
        return self.combined2hidden(combined), (attention_scores, attention_distribution)


class DotProductAttention(Attention):
    def score(self, output: Tensor, context: Tensor) -> Tensor:
        #  | [batch_size, output_sequence_length, hidden_size]
        #  @ [batch_size, hidden_size,            context_sequence_length]
        # -> [batch_size, output_sequence_length, context_sequence_length]
        return torch.bmm(output, context.transpose(1, 2))


class MultiplicativeAttention(Attention):
    def score(self, output: Tensor, context: Tensor) -> Tensor:
        raise NotImplementedError


class AdditiveAttention(Attention):
    def score(self, output: Tensor, context: Tensor) -> Tensor:
        raise NotImplementedError
