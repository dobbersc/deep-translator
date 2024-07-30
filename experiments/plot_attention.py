from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import Tensor

from translator.datasets import EuroparlCorpus
from translator.models import Translator
from translator.utils.random import set_seed
from translator.utils.torch import detect_device

from experiments import RESULTS_DIRECTORY


def plot_attention(
    source_tokens: Sequence[str],
    target_tokens: Sequence[str],
    attention_matrix: torch.Tensor,
    out: Path,
) -> None:
    fig, ax = plt.subplots()

    ax_img = ax.matshow(attention_matrix.detach().cpu().numpy().T, vmin=0, vmax=1)
    plt.colorbar(ax_img)

    plt.tick_params(axis="both", which="major", labelsize=7)
    ax.set_xticks(np.arange(len(target_tokens)), labels=target_tokens, rotation=65)
    ax.set_yticks(np.arange(len(source_tokens)), labels=source_tokens)

    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()

    ax.set_title("Attention Matrix")

    plt.tight_layout()
    plt.savefig(out)
    plt.close(fig)


def translate_and_plot_attention(translator: Translator, source: str, target: str, out: Path) -> None:
    source_tokens: Sequence[str] = translator.source_tokenizer(source)
    target_tokens: Sequence[str] = translator.target_tokenizer(target)

    vectorized_source: Tensor = torch.tensor(
        [translator.source_language.encode(source_tokens)],
        dtype=torch.long,
        device=translator.device,
    )
    vectorized_target: Tensor = torch.tensor(
        [translator.target_language.encode(target_tokens)],
        dtype=torch.long,
        device=translator.device,
    )

    _, (_, attention_distributions) = translator.forward(
        vectorized_source,
        vectorized_target,
        teacher_forcing_ratio=1,
        return_attention=True,
    )

    plot_attention(
        source_tokens=["<START>", *source_tokens],
        target_tokens=[*target_tokens, "<END>"],
        attention_matrix=attention_distributions.squeeze(dim=0),
        out=out,
    )


def main() -> None:
    set_seed(42)

    device: torch.device = detect_device()
    translator: Translator = Translator.load(
        RESULTS_DIRECTORY / "models" / "de2en" / "word_word2vec_embeddings_with_attention" / "model.pt",
        map_location=device,
    )
    translator.to(device)

    out_directory: Path = RESULTS_DIRECTORY / "plots"
    out_directory.mkdir(exist_ok=True, parents=True)

    corpus: EuroparlCorpus = EuroparlCorpus.load(translator.source_language.name, translator.target_language.name)
    corpus = corpus.downsample(0.1)
    _, _, test_split = corpus.split()

    # Select representative data points from the test data.
    for index in (37004, 38028):
        translate_and_plot_attention(
            translator,
            source=test_split[index].source,
            target=test_split[index].target,
            out=out_directory / f"attention_plot_{index}.pdf",
        )


if __name__ == "__main__":
    main()
