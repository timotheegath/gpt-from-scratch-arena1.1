import math
from dataclasses import dataclass
from typing import Any, cast

import datasets  # type: ignore
from beartype import beartype as typechecker
from jaxtyping import Float, Int, jaxtyped
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from transformer_lens.utilities.tokenize_utils import tokenize_and_concatenate
from transformers import PreTrainedTokenizerBase

# ruff: noqa: F722


@dataclass
class TransformerTrainingArgs:
    batch_size: int = 32
    epochs: int = 10
    max_steps_per_epoch: int = 500
    lr: float = 1e-3
    weight_decay: float = 1e-2
    wandb_project: str | None = "day1-demotransformer"
    wandb_name: str | None = None


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


def load_dataset(
    tokenizer: PreTrainedTokenizerBase, context_size: int, training_params: TransformerTrainingArgs
) -> tuple[DataLoader[Any], DataLoader[Any]]:
    dataset: datasets.Dataset = datasets.load_dataset("roneneldan/TinyStories", split="train")
    tokenized_dataset = tokenize_and_concatenate(
        dataset,
        tokenizer,
        streaming=False,
        max_length=context_size,
        column_name="text",
        add_bos_token=True,
        num_proc=12,
    )
    assert isinstance(tokenized_dataset, datasets.Dataset)
    dataset_dict = tokenized_dataset.train_test_split(test_size=1000)
    train_loader = DataLoader(
        cast("TorchDataset[Any]", dataset_dict["train"]),
        batch_size=training_params.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        cast("TorchDataset[Any]", dataset_dict["test"]),
        batch_size=training_params.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, test_loader  # Dataset of shape [batch, sequence_length (max_ctx_size)]
