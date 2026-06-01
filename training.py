import math

from beartype import beartype as typechecker
from jaxtyping import Float, Int, jaxtyped
from torch import Tensor

# ruff: noqa: F722


@jaxtyped(typechecker=typechecker)
def get_log_probs(
    logits: Float[Tensor, "batch posn d_vocab"], tokens: Int[Tensor, "batch posn"], d_vocab: int
) -> Float[Tensor, "batch posn-1"]:
    log_probs = logits.log_softmax(dim=-1)
    # Get logprobs the first seq_len-1 predictions (so we can compare them with the actual next tokens)
    log_probs_for_tokens = (
        log_probs[:, :-1].gather(dim=-1, index=tokens[:, 1:].unsqueeze(-1)).squeeze(-1)
    )
    print(f"Avg cross entropy loss: {-log_probs_for_tokens.mean():.4f}")
    print(f"Avg cross entropy loss for uniform distribution: {math.log(d_vocab):4f}")
    print(f"Avg probability assigned to correct token: {log_probs_for_tokens.exp().mean():4f}")
    return log_probs_for_tokens
