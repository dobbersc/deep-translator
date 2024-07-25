from typing import Any, Literal

from torch import Tensor

from translator.nn.decoder import DecoderLSTM
from translator.nn.encoder import EncoderLSTM
from translator.nn.module import Module


class Seq2Seq(Module):
    def __init__(self, encoder: EncoderLSTM, decoder: DecoderLSTM) -> None:
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, sources: Tensor, targets: Tensor, teacher_forcing_ratio: float = 0.5) -> Tensor:
        encoder_hidden_states, encoder_hidden_and_cell = self.encoder(sources)
        log_probabilities: Tensor = self.decoder(
            targets,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_and_cell=encoder_hidden_and_cell,
            teacher_forcing_ratio=teacher_forcing_ratio,
        )
        return log_probabilities

    def generate_greedy(self, sources: Tensor, max_length: int = 512) -> Tensor:
        encoder_hidden_states, encoder_hidden_and_cell = self.encoder(sources)
        return self.decoder.decode_greedy(encoder_hidden_states, encoder_hidden_and_cell, max_length=max_length)

    def generate_sampled(self, sources: Tensor, temperature: float = 0.8, max_length: int = 512) -> Tensor:
        encoder_hidden_states, encoder_hidden_and_cell = self.encoder(sources)
        return self.decoder.decode_sampled(
            encoder_hidden_states,
            encoder_hidden_and_cell,
            temperature=temperature,
            max_length=max_length,
        )

    def generate_beam_search(self, sources: Tensor, max_length: int = 512) -> Tensor:
        raise NotImplementedError

    def generate(
        self,
        sources: Tensor,
        method: Literal["greedy", "sampled", "beam-search"] = "sampled",
        **kwargs: Any,
    ) -> Tensor:
        if method == "greedy":
            return self.generate_greedy(sources, **kwargs)
        if method == "sampled":
            return self.generate_sampled(sources, **kwargs)
        if method == "beam-search":
            return self.generate_beam_search(sources, **kwargs)
        raise ValueError
