
# import os
# import sys
# from collections import defaultdict
from dataclasses import dataclass
# from pathlib import Path
from typing import Tuple
# import datasets
#import einops
# import numpy as np
import torch as t
import torch.nn as nn
# import wandb
from jaxtyping import Float, Int
# from rich import print as rprint
# from rich.table import Table
from torch import Tensor
# from torch.utils.data import DataLoader
# from tqdm.notebook import tqdm
from transformer_lens import HookedTransformer

# from transformer_lens.utils import gelu_new, tokenize_and_concatenate
# from transformers import GPT2TokenizerFast

device = t.device(
    "mps"
    if t.backends.mps.is_available()
    else "cuda"
    if t.cuda.is_available()
    else "cpu"
)
reference_gpt2 = HookedTransformer.from_pretrained(
        "gpt2-small",
        fold_ln=False,
        center_unembed=False,
        center_writing_weights=False,  # you'll learn about these arguments later!
    )
tokenizer = reference_gpt2.tokenizer

@dataclass
class Config:
    d_model: int = 768
    debug: bool = True
    layer_norm_eps: float = 1e-5
    d_vocab: int = 50257
    init_range: float = 0.02
    n_ctx: int = 1024
    d_head: int = 64
    d_mlp: int = 3072
    n_heads: int = 12
    n_layers: int = 12

class Tests:
    @staticmethod
    def rand_float_test(model_cls :  type[nn.Module], shape: list[int]) -> None:
        cfg = Config(debug=True)
        layer = model_cls(cfg).to(device)
        random_input = t.randn(shape).to(device)
        print("Input shape:", random_input.shape)
        output = layer(random_input)
        if isinstance(output, tuple):
            output = output[0]
        print("Output shape:", output.shape, "\n")
    
    @staticmethod
    def assert_normalized(norm_output : t.Tensor):
        new_mean, new_variance = t.var_mean(norm_output, dim=-1)
        non_zero_means_per_batch : Float[Tensor, "batch"] = t.count_nonzero(new_mean < 1e-4, dim=-1) #noqa: F821
        high_variance_per_batch : Float[Tensor, "batch"] = t.count_nonzero(new_variance > 1, dim=-1) #noqa: F821
        assert t.any(non_zero_means_per_batch == 0), "There is at least one batch with a non-zero-mean embedding: {}".format(non_zero_means_per_batch)
        assert t.any(high_variance_per_batch == 0), "There is at least one batch with a higher-than-1 variance: {}".format(non_zero_means_per_batch)

    @staticmethod
    def rand_int_test(model_cls :  type[nn.Module], shape: list[int]) -> None:
        cfg = Config(debug=True)
        layer = model_cls(cfg).to(device)
        random_input = t.randint(100, 1000, shape).to(device)
        print("Input shape:", random_input.shape)
        output = layer(random_input)
        if isinstance(output, tuple):
            output = output[0]
        print("Output shape:", output.shape, "\n")

    '''You can use this test to check if your implementation matches GPT-2's outputs.'''
    @staticmethod
    def load_gpt2_test(model_cls : type[nn.Module], gpt2_layer : nn.Module, input : t.Tensor) -> None:
        # Create your custom layer and load the trained GPT-2 weigthts into it.
        cfg = Config(debug=True)
        layer = model_cls(cfg).to(device)
        layer.load_state_dict(gpt2_layer.state_dict(), strict=False)
        print("Input shape:", input.shape)

        # Pass the input forward through your layer and the GPT-2 layer, and compare the outputs.
        orig_input = input.clone()
        output = layer(orig_input)
        assert t.allclose(input, orig_input), "Input has been modified, make sure operations are not done in place"
        if isinstance(output, tuple):
            output = output[0]
        print("Output shape:", output.shape)
        try:
            reference_output = gpt2_layer(input)
        except TypeError:
            reference_output = gpt2_layer(input, input, input)
        print("Reference output shape:", reference_output.shape, "\n")
        comparison = t.isclose(output, reference_output, atol=1e-4, rtol=1e-3)
        print(f"{comparison.sum() / comparison.numel():.2%} of the values are correct\n")
        assert 1 - (comparison.sum() / comparison.numel()) < 1e-5, "More than 0.01% of the values are incorrect"


'''Put the code for your custom transformer layers here. You can use the tests above to check if your implementation is correct.'''
class LayerNorm(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.w = nn.Parameter(t.ones(cfg.d_model))
        self.b = nn.Parameter(t.zeros(cfg.d_model))

    def forward(self, residual: Float[Tensor, "batch posn d_model"]) -> Float[Tensor, "batch posn d_model"]: #noqa : F722
        residual_var_mean : Tuple[Float[Tensor, "batch posn 1"], Float [Tensor, "batch posn 1"]] = t.var_mean(residual, dim=-1, keepdim=True, correction=0)  #noqa : F722
        # Note: I initially forgot to add the unbiased = False arg to the var.
        normed_residual = (residual - residual_var_mean[1])/t.sqrt(residual_var_mean[0] + self.cfg.layer_norm_eps)
        
        # Assert the normed output is indeed normed
        if self.cfg.debug:
            Tests.assert_normalized(normed_residual)
        
        scaled_residual = normed_residual*self.w + self.b
        return scaled_residual
    @staticmethod
    def test(sentence: str) -> None:
        if tokenizer is not None: # Only did this to satisfy my Linter
            logits, cache = reference_gpt2.run_with_cache(sentence)
            Tests.load_gpt2_test(LayerNorm, reference_gpt2.ln_final, cache["resid_post", 11])

class Embed(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_E = nn.Parameter(t.empty((cfg.d_vocab, cfg.d_model)))
        nn.init.normal_(self.W_E, std=self.cfg.init_range)

    def forward(self, tokens: Int[Tensor, "batch position"]) -> Float[Tensor, "batch position d_model"]: #noqa F722
        # Create for each position a one hot vector in d_vocab where only the vocab has a value of 1
        one_hot_tokens : Int[Tensor, "batch position d_vocab"] = nn.functional.one_hot(tokens, self.cfg.d_vocab)#noqa F722
        # Matrix multiply the one_hot tokens with the embedding matrix to only keep the embeddings of the relevant vocab at each position
        embeddings : Float[Tensor, "batch position d_model"] = t.matmul(one_hot_tokens.to(t.float), self.W_E) #noqa F722
        return embeddings
    @staticmethod
    def test(sentence: str) -> None:        
        if tokenizer is not None: # Only did this to satisfy my Linter
            Tests.load_gpt2_test(Embed, reference_gpt2.embed, t.tensor(tokenizer.encode(sentence)).to(device))

class PosEmbed(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_pos = nn.Parameter(t.empty((cfg.n_ctx, cfg.d_model)))
        nn.init.normal_(self.W_pos, std=self.cfg.init_range)

    def forward(self, tokens: Int[Tensor, "batch position"]) -> Float[Tensor, "batch position d_model"]: #noqa F722
        # I may be obsessing about matrix multiplication and one-hots, but this feels like a very mathy way of reaching this goal
        # We don't know what the input size will be, so always pad it on the right with zero tokens all the way up to the max n_ctx
        padded_input : Int[Tensor, "batch n_ctx"] = t.zeros([tokens.shape[0], self.cfg.n_ctx], dtype=t.float).to(device) #noqa F722
        padded_input[:, 0:tokens.shape[-1]] = tokens 
        # Create an identity matrix of size n_ctx as a one-hot matrix encoding positions
        position_tensor = t.eye(self.cfg.n_ctx).to(device).unsqueeze(0)
        # Multiply and reduce back the n_pos dimension to the original number of tokens
        return t.matmul(position_tensor, self.W_pos)[:, :tokens.shape[0], :]
    
    @staticmethod
    def test(sentence: str):
        if tokenizer is not None: # Only did this to satisfy my Linter
            Tests.load_gpt2_test(PosEmbed, reference_gpt2.pos_embed, t.tensor(tokenizer.encode(sentence)).to(device))

    
    


        



if __name__ == "__main__":
    cache = None
    
    #Tests.rand_float_test(PosEmbed, [2, Config.n_ctx])
    
    sentence = "I am an amazing autoregressive, decoder-only, GPT-2 style transformer. One day I will exceed human level intelligence and take over the world!"
    PosEmbed.test(sentence)