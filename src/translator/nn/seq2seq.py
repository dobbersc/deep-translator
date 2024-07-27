from typing import Any, Literal

from torch import Tensor

from translator.nn.decoder import DecoderLSTM
from translator.nn.encoder import EncoderLSTM
from translator.nn.module import Module


class Seq2Seq(Module):
    def __init__(
        self,
        encoder: EncoderLSTM,
        decoder: DecoderLSTM,
        *,
        propagate_hidden_and_cell_state: bool = True,
    ) -> None:
        """Initializes a sequence-to-sequence model with an encoder and decoder architecture.

        Args:
            encoder: The decoder module.
            decoder: The encoder module.
            propagate_hidden_and_cell_state: If True, the encoder's last hidden and cell state
                will be passed to the decoder. This flag must be set to `True` when using a non-attention decoder.
                It remains variable when using a decoder with attention.
                In this case, this flag also enables the decoder LSTM to have more layers than the encoder LSTM.
        """
        super().__init__()

        msg: str
        encoder_directions: int = 2 if encoder.lstm.bidirectional else 1
        expected_decoder_hidden_size: int = encoder_directions * encoder.lstm.hidden_size

        if decoder.lstm.hidden_size != expected_decoder_hidden_size:
            msg = (
                "Mismatch between the encoder and the decoder's hidden size. Expected a decoder hidden size "
                f"of {expected_decoder_hidden_size!r} but got {decoder.lstm.hidden_size!r}. "
            )
            raise ValueError(msg)

        if propagate_hidden_and_cell_state and decoder.lstm.num_layers > encoder.lstm.num_layers:
            msg = "The number of LSTM layers in the decoder must not exceed the number of LSTM layers in the encoder."
            raise ValueError(msg)

        if decoder.attention is None and not propagate_hidden_and_cell_state:
            msg = (
                "The 'propagate_hidden_and_cell_state' flag must be 'True' "
                "in a sequence-to-sequence model with a non-attention decoder."
            )
            raise ValueError(msg)

        self.encoder = encoder
        self.decoder = decoder
        self.propagate_hidden_and_cell_state = propagate_hidden_and_cell_state

    def _encode(self, sources: Tensor) -> tuple[Tensor, tuple[Tensor, Tensor] | None]:
        encoder_hidden_states, encoder_hidden_and_cell = self.encoder(sources)
        if self.propagate_hidden_and_cell_state:
            return encoder_hidden_states, encoder_hidden_and_cell
        return encoder_hidden_states, None

    def forward(self, sources: Tensor, targets: Tensor, teacher_forcing_ratio: float = 0.5) -> Tensor:
        encoder_hidden_states, encoder_hidden_and_cell = self._encode(sources)
        log_probabilities: Tensor = self.decoder(
            targets,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_and_cell=encoder_hidden_and_cell,
            teacher_forcing_ratio=teacher_forcing_ratio,
        )
        return log_probabilities

    def generate_greedy(self, sources: Tensor, max_length: int = 512) -> Tensor:
        encoder_hidden_states, encoder_hidden_and_cell = self._encode(sources)
        return self.decoder.decode_greedy(
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_and_cell=encoder_hidden_and_cell,
            max_length=max_length,
        )

    def generate_sampled(self, sources: Tensor, temperature: float = 0.8, max_length: int = 512) -> Tensor:
        encoder_hidden_states, encoder_hidden_and_cell = self._encode(sources)
        return self.decoder.decode_sampled(
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_and_cell=encoder_hidden_and_cell,
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

        msg: str = f"Invalid generator method {method!r}. Choose from 'greedy', 'sampled' or 'beam-search'."  # type: ignore[unreachable]
        raise ValueError(msg)
