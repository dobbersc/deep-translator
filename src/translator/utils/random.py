import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Sets the global seed for the builtin random, numpy and torch modules.

    Args:
        seed: The desired seed.
    """
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)
