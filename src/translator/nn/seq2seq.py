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
            targets, encoder_hidden_states, encoder_hidden_and_cell, teacher_forcing_ratio,
        )
        return log_probabilities

    def generate(self, sources: Tensor) -> Tensor:
        raise NotImplementedError
