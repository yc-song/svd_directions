from typing import List, Union

import numpy as np
import torch

Tensor = Union[np.ndarray, torch.Tensor]


def init(
    tokens: List[str], attention: Tensor, pos_logits: Tensor, neg_logits: Tensor, head_labels=None
):
    """Visualize the attention patterns and logit attribution for multiple attention heads.

    Args:
      tokens: a list of strings representing tokens
      attention: A [N, N, H] array representing attention probabilities,
        where N is the number of tokens and H is the number of heads
        (or analogous value like number of NMF factors).

        Attention weights are expected to be in [0, 1].

      head_labels: human readable labels for heads. Optional.

    """
    assert (
        len(tokens) == attention.shape[0]
    ), "tokens and activations must be same length"
    assert (
        attention.shape[0] == attention.shape[1]
    ), "first two dimensions of attention must be equal"
    assert attention.ndim == 3, "attention must be 3D"
    assert (
        len(tokens) == pos_logits.shape[0]
    ), "tokens and pos_logits must be same length"
    assert (
        pos_logits.shape[0] == pos_logits.shape[1]
    ), "first two dimensions of pos_logits must be equal"
    assert pos_logits.ndim == 3, "pos_logits must be 3D"
    assert (
        len(tokens) == neg_logits.shape[0]
    ), "tokens and neg_logits must be same length"
    assert (
        neg_logits.shape[0] == neg_logits.shape[1]
    ), "first two dimensions of neg_logits must be equal"
    assert neg_logits.ndim == 3, "neg_logits must be 3D"
    if head_labels is not None:
        assert (
            len(head_labels) == attention.shape[-1]
        ), "head_labels must correspond to number of attention heads"
    return
