import math
from dataclasses import dataclass
from typing import Any, cast

import datasets  # type: ignore
import numpy as np
import torch as t
from beartype import beartype as typechecker
from jaxtyping import Float, Int, jaxtyped
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from tqdm import tqdm
from transformer_lens.utilities.tokenize_utils import tokenize_and_concatenate
from transformers import PreTrainedTokenizerBase

import part1_transformer_from_scratch.solutions as solutions
import wandb
from custom_transformer import DemoTransformer

# ruff: noqa: F722
device = t.device(
    "mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu"
)


@dataclass
class TransformerTrainingArgs:
    batch_size: int = 32
    epochs: int = 10
    max_steps_per_epoch: int = 500
    lr: float = 1e-3
    weight_decay: float = 1e-2
    wandb_project: str | None = "day1-demotransformer"
    wandb_name: str | None = None


class TransformerTrainer:
    def __init__(self, args: TransformerTrainingArgs, model: DemoTransformer):
        super().__init__()
        self.model = model
        self.args = args
        self.sampler = solutions.TransformerSampler(self.model, self.model.tokenizer)
        self.optimizer = t.optim.AdamW(
            self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        self.step = 0

        self.train_loader, self.test_loader = load_dataset(self.model.tokenizer, 100, args)

    def training_step(self, batch: dict[str, Int[Tensor, "batch seq"]]) -> Float[Tensor, ""]:
        """
        Calculates the loss on the tokens in the batch, performs a gradient update step, and logs the loss.

        Remember that `batch` is a dictionary with the single key 'tokens'.
        """
        tokens = batch["tokens"].to(device)
        logits = self.model(tokens)
        loss = -self.loss(logits, tokens, self.model.cfg.d_vocab).mean()
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.step += 2
        wandb.log({"train_loss": loss}, step=self.step)
        return loss

    @t.inference_mode()
    def evaluate(self) -> float:
        """
        Evaluate the model on the test set and return the accuracy.
        """
        self.model.eval()
        total_correct = 0
        total_evaluations = 0
        for i, batch in enumerate(self.test_loader):
            tokens = batch["tokens"].to(device)
            logits: Float[Tensor, "batch pos_n d_vocab"] = self.model(tokens)
            output_tokens = logits.argmax(-1)
            total_correct += t.sum(output_tokens[:, :-1] == tokens[:, 1:]).item()
            total_evaluations += tokens.size(0) * tokens.size(1) - 1
        self.model.train()
        accuracy = total_correct / total_evaluations
        wandb.log({"accuracy": accuracy}, step=self.step)
        return accuracy

    def train(self) -> None:
        """
        Trains the model, for `self.args.epochs` epochs. Also handles wandb initialisation, and early stopping
        for each epoch at `self.args.max_steps_per_epoch` steps.
        """
        wandb.init(project=self.args.wandb_project, name=self.args.wandb_name, config=self.args)
        accuracy = np.nan

        progress_bar = tqdm(total=self.args.max_steps_per_epoch * self.args.epochs)

        for epoch in range(self.args.epochs):
            for i, batch in enumerate(self.train_loader):
                loss = self.training_step(batch)
                progress_bar.update()
                progress_bar.set_description(
                    f"Epoch {epoch + 1}, loss: {loss:.3f}, accuracy: {accuracy:.3f}"
                )
                if i >= self.args.max_steps_per_epoch:
                    break

            accuracy = self.evaluate()
            sample_text = self.sampler.sample("Once upon a time", max_tokens_generated=50)
            print(sample_text)

        wandb.finish()

    @staticmethod
    def loss(
        logits: Float[Tensor, "batch posn d_vocab"], tokens: Int[Tensor, "batch posn"], d_vocab: int
    ) -> Float[Tensor, "batch posn-1"]:
        log_probs = logits.log_softmax(dim=-1)
        # Get logprobs the first seq_len-1 predictions (so we can compare them with the actual next tokens)
        log_probs_for_tokens = (
            log_probs[:, :-1].gather(dim=-1, index=tokens[:, 1:].unsqueeze(-1)).squeeze(-1)
        )
        # print(f"Avg cross entropy loss: {-log_probs_for_tokens.mean():.4f}")
        # print(f"Avg cross entropy loss for uniform distribution: {math.log(d_vocab):4f}")
        # print(f"Avg probability assigned to correct token: {log_probs_for_tokens.exp().mean():4f}")
        return log_probs_for_tokens


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
        generator=t.Generator(device="cpu"),
    )
    test_loader = DataLoader(
        cast("TorchDataset[Any]", dataset_dict["test"]),
        batch_size=training_params.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        generator=t.Generator(device="cpu"),
    )

    return train_loader, test_loader  # Dataset of shape [batch, sequence_length (max_ctx_size)]
