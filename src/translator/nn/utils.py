from typing import Any

import torch
from gensim.models import KeyedVectors
from torch import Tensor


def build_embedding_layer(
    num_embeddings: int,
    embedding_size: int,
    *,
    pretrained_embeddings: KeyedVectors | Tensor | None = None,
    freeze_pretrained_embeddings: bool = True,
    padding_index: int | None = None,
    **kwargs: Any,
) -> torch.nn.Embedding:
    embedding: torch.nn.Embedding
    if pretrained_embeddings is None:
        embedding = torch.nn.Embedding(num_embeddings, embedding_size, padding_idx=padding_index, **kwargs)
    else:
        if isinstance(pretrained_embeddings, KeyedVectors):
            pretrained_embeddings = torch.tensor(pretrained_embeddings.vectors, dtype=torch.float)

        embedding = torch.nn.Embedding.from_pretrained(
            pretrained_embeddings,
            freeze=freeze_pretrained_embeddings,
            padding_idx=padding_index,
            **kwargs,
        )

        if embedding_size != embedding.embedding_dim:
            msg = (
                f"Embedding size mismatch of pretrained embeddings: "
                f"expected {embedding_size!r}, but got {embedding.embedding_dim!r}."
            )
            raise ValueError(msg)

    return embedding
