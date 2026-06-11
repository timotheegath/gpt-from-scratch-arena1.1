from dataclasses import dataclass  # noqa: I001
from typing import cast

import numpy as np
import torch as t
import torch.nn as nn

from torch import Tensor
from beartype import beartype as typechecker
from jaxtyping import Float, Int, jaxtyped
from tqdm import tqdm
from transformer_lens import HookedTransformer
from transformer_lens.utilities.activation_functions import gelu_new
from transformers import PreTrainedTokenizerBase
from transformers.models.gpt2.tokenization_gpt2 import GPT2Tokenizer as GPT2TokenizerFast
# from training import get_log_probs

# ruff: noqa: F722, F821

device = t.device(
    "mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu"
)
""" reference_gpt2 = HookedTransformer.from_pretrained(
    "gpt2-small",
    fold_ln=False,
    center_unembed=False,
    center_writing_weights=False,  # you'll learn about these arguments later!
)
tokenizer = reference_gpt2.tokenizer """


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
    def rand_float_test(model_cls: type[nn.Module], shape: list[int]) -> None:
        cfg = Config(debug=True)
        layer = model_cls(cfg).to(device)
        random_input = t.randn(shape).to(device)
        print("Input shape:", random_input.shape)
        output = layer(random_input)
        if isinstance(output, tuple):
            output = output[0]
        print("Output shape:", output.shape, "\n")

    @staticmethod
    def assert_normalized(norm_output: t.Tensor) -> None:
        new_mean, new_variance = t.var_mean(norm_output, dim=-1)
        non_zero_means_per_batch: Float[Tensor, "batch"] = t.count_nonzero(new_mean < 1e-4, dim=-1)  # noqa: UP037, F821
        high_variance_per_batch: Float[Tensor, "batch"] = t.count_nonzero(new_variance > 1, dim=-1)  # noqa: UP037, F821
        assert t.any(non_zero_means_per_batch == 0), (
            f"There is at least one batch with a non-zero-mean embedding: {non_zero_means_per_batch}"
        )
        assert t.any(high_variance_per_batch == 0), (
            f"There is at least one batch with a higher-than-1 variance: {non_zero_means_per_batch}"
        )

    @staticmethod
    def rand_int_test(model_cls: type[nn.Module], shape: list[int]) -> None:
        cfg = Config(debug=True)
        layer = model_cls(cfg).to(device)
        random_input = t.randint(100, 1000, shape).to(device)
        print("Input shape:", random_input.shape)
        output = layer(random_input)
        if isinstance(output, tuple):
            output = output[0]
        print("Output shape:", output.shape, "\n")

    """You can use this test to check if your implementation matches GPT-2's outputs."""

    @staticmethod
    def load_gpt2_test(model_cls: type[nn.Module], gpt2_layer: nn.Module, input: t.Tensor) -> None:
        # Create your custom layer and load the trained GPT-2 weigthts into it.
        cfg = Config(debug=True)
        layer = model_cls(cfg).to(device)
        layer.load_state_dict(gpt2_layer.state_dict(), strict=False)
        print("Input shape:", input.shape)

        # Pass the input forward through your layer and the GPT-2 layer, and compare the outputs.
        orig_input = input.clone()
        output = layer(orig_input)
        assert t.allclose(input, orig_input), (
            "Input has been modified, make sure operations are not done in place"
        )
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
        assert 1 - (comparison.sum() / comparison.numel()) < 1e-5, (
            "More than 0.01% of the values are incorrect"
        )


"""Put the code for your custom transformer layers here. You can use the tests above to check if your implementation is correct."""


class LayerNorm(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.w = nn.Parameter(t.ones(cfg.d_model), requires_grad=True)
        self.b = nn.Parameter(t.zeros(cfg.d_model), requires_grad=True)

    def forward(
        self, residual: Float[Tensor, "batch posn d_model"]
    ) -> Float[Tensor, "batch posn d_model"]:
        residual_var_mean: tuple[Float[Tensor, "batch posn 1"], Float[Tensor, "batch posn 1"]] = (
            t.var_mean(residual, dim=-1, keepdim=True, correction=0)
        )
        # Note: I initially forgot to add the unbiased = False arg to the var.
        normed_residual = (residual - residual_var_mean[1]) / t.sqrt(
            residual_var_mean[0] + self.cfg.layer_norm_eps
        )

        # Assert the normed output is indeed normed
        if self.cfg.debug:
            Tests.assert_normalized(normed_residual)

        scaled_residual = normed_residual * self.w + self.b
        return scaled_residual

    @staticmethod
    def test(
        sentence: str, tokenizer: PreTrainedTokenizerBase, reference_gpt2: HookedTransformer
    ) -> None:
        if tokenizer is not None:  # Only did this to satisfy my Linter
            logits, cache = reference_gpt2.run_with_cache(sentence)
            Tests.load_gpt2_test(LayerNorm, reference_gpt2.ln_final, cache["resid_post", 11])


class Embed(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_E = nn.Parameter(t.empty((cfg.d_vocab, cfg.d_model)), requires_grad=True)
        nn.init.normal_(self.W_E, std=self.cfg.init_range)

    def forward(
        self, tokens: Int[Tensor, "batch position"]
    ) -> Float[Tensor, "batch position d_model"]:  # noqa F722
        # Create for each position a one hot vector in d_vocab where only the vocab has a value of 1
        one_hot_tokens: Int[Tensor, "batch position d_vocab"] = nn.functional.one_hot(
            tokens.to(t.long), self.cfg.d_vocab
        )
        # Matrix multiply the one_hot tokens with the embedding matrix to only keep the embeddings of the relevant vocab at each position
        embeddings: Float[Tensor, "batch position d_model"] = t.matmul(
            one_hot_tokens.to(t.float), self.W_E
        )
        return embeddings

    @staticmethod
    def test(
        sentence: str, tokenizer: PreTrainedTokenizerBase, reference_gpt2: HookedTransformer
    ) -> None:
        if tokenizer is not None:  # Only did this to satisfy my Linter
            Tests.load_gpt2_test(
                Embed, reference_gpt2.embed, t.tensor(tokenizer.encode(sentence)).to(device)
            )


class PosEmbed(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_pos = nn.Parameter(t.empty((cfg.n_ctx, cfg.d_model)), requires_grad=True)
        nn.init.normal_(self.W_pos, std=self.cfg.init_range)

    def forward(
        self, tokens: Int[Tensor, "batch position"]
    ) -> Float[Tensor, "batch position d_model"]:  # noqa F722

        batch, seq_len = tokens.shape
        pos_embed = self.W_pos[:seq_len].unsqueeze(0).expand(batch, -1, -1)
        return pos_embed

    @staticmethod
    def test(
        sentence: str, tokenizer: PreTrainedTokenizerBase, reference_gpt2: HookedTransformer
    ) -> None:
        if tokenizer is not None:  # Only did this to satisfy my Linter
            Tests.load_gpt2_test(
                PosEmbed, reference_gpt2.pos_embed, t.tensor(tokenizer.encode(sentence)).unsqueeze(0).to(device)
            )

    @staticmethod
    def test_with_random() -> None:
        Tests.rand_int_test(PosEmbed, [2, 4])


class Attention(nn.Module):
    IGNORE: Float[Tensor, ""]

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_Q = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)), requires_grad=True)
        self.W_K = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)), requires_grad=True)
        self.W_V = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)), requires_grad=True)
        self.W_O = nn.Parameter(t.empty((cfg.n_heads, cfg.d_head, cfg.d_model)), requires_grad=True)
        self.b_Q = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)), requires_grad=True)
        self.b_K = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)), requires_grad=True)
        self.b_V = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)), requires_grad=True)
        self.b_O = nn.Parameter(t.zeros(cfg.d_model), requires_grad=True)
        nn.init.normal_(self.W_Q, std=self.cfg.init_range)
        nn.init.normal_(self.W_K, std=self.cfg.init_range)
        nn.init.normal_(self.W_V, std=self.cfg.init_range)
        nn.init.normal_(self.W_O, std=self.cfg.init_range)
        self.register_buffer("IGNORE", t.tensor(float("-inf"), dtype=t.float32, device=device))

    @jaxtyped(typechecker=typechecker)
    def forward(
        self, normalized_resid_pre: Float[Tensor, "batch posn d_model"]
    ) -> Float[Tensor, "batch posn d_model"]:
        # 1. Compute query key vectors
        q: Float[Tensor, "batch n_heads posn d  d_head"] = t.matmul(
            normalized_resid_pre.unsqueeze(1), self.W_Q.unsqueeze(0)
        ) + self.b_Q.unsqueeze(0).unsqueeze(2)
        k: Float[Tensor, "batch n_heads posn d  d_head"] = t.matmul(
            normalized_resid_pre.unsqueeze(1), self.W_K.unsqueeze(0)
        ) + self.b_K.unsqueeze(0).unsqueeze(2)
        # 2. In parallel, compute the value vectors
        v: Float[Tensor, "batch n_heads posn d  d_head"] = t.matmul(
            normalized_resid_pre.unsqueeze(1), self.W_V.unsqueeze(0)
        ) + self.b_V.unsqueeze(0).unsqueeze(2)
        # 3. Create the full attention pattern (no masking, softmax yet)
        qk: Float[Tensor, "batch n_heads posn d posn_d"] = t.matmul(q, k.transpose(2, 3))
        # 4. Scale the attention matrix to avoid vanishing gradients
        qk = qk / t.sqrt(Tensor([self.cfg.d_head]).to(device).to(t.float))
        # 5. Mask key indexes higher than query indexes to force the model to look back only..
        qk_masked: Float[Tensor, "batch n_heads posn d posn_d"] = self.apply_causal_mask(qk)
        # 6. Convert into a probability distribution for each query row, along the key column dimension:
        qk_p: Float[Tensor, "batch n_heads posn d posn_d"] = t.softmax(qk_masked, 3)
        # 7. Do the weighted average of the value vectors using the key vectors
        v_avg: Float[Tensor, "batch n_heads posn d d_head"] = t.matmul(qk_p, v)
        # 8. Linear layer with scale, sum the heads and add bias to finish before the output of the block:
        o: Float[Tensor, "batch posn d d_model"] = t.matmul(v_avg, self.W_O).sum(dim=1) + self.b_O

        # ave_image(QK_p[0,0,:,:], 'GREY_img.png')

        return o

    def apply_causal_mask(
        self,
        attn_scores: Float[Tensor, "batch n_heads query_pos key_pos"],
    ) -> Float[Tensor, "batch n_heads query_pos key_pos"]:
        """
        Applies a causal mask to attention scores, and returns masked scores.
        """
        all_ones = t.ones(attn_scores.size(-2), attn_scores.size(-1), device=attn_scores.device)
        mask = t.triu(all_ones, diagonal=1).bool()
        return attn_scores.masked_fill(mask, self.IGNORE)

    @staticmethod
    def test(
        sentence: str, tokenizer: PreTrainedTokenizerBase, reference_gpt2: HookedTransformer
    ) -> None:
        if tokenizer is not None:
            logits, cache = reference_gpt2.run_with_cache(sentence)
            Tests.load_gpt2_test(
                Attention,
                cast("nn.Module", reference_gpt2.blocks[0].attn),
                cache["normalized", 0, "ln1"],
            )

    def test_causal_mask(self) -> None:
        input = t.rand(
            [
                2,
                self.cfg.d_head,
                self.cfg.d_model // self.cfg.d_head,
                self.cfg.d_model // self.cfg.d_head,
            ]
        ).to(device)
        print("Input shape:", input.shape)
        output = self.apply_causal_mask(input)
        if isinstance(output, tuple):
            output = output[0]
        print("Output shape:", output.shape, "\n")
        print(output[0, 0, :, :])


class MLP(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_in = nn.Parameter(t.empty((cfg.d_model, cfg.d_mlp)), requires_grad=True)
        self.W_out = nn.Parameter(t.empty((cfg.d_mlp, cfg.d_model)), requires_grad=True)
        self.b_in = nn.Parameter(t.zeros(cfg.d_mlp), requires_grad=True)
        self.b_out = nn.Parameter(t.zeros(cfg.d_model), requires_grad=True)
        nn.init.normal_(self.W_in, std=self.cfg.init_range)
        nn.init.normal_(self.W_out, std=self.cfg.init_range)

    def forward(
        self, normalized_resid_mid: Float[Tensor, "batch posn d_model"]
    ) -> Float[Tensor, "batch posn d_model"]:

        hidden_space = gelu_new(t.matmul(normalized_resid_mid, self.W_in.unsqueeze(0)) + self.b_in)
        out = t.matmul(hidden_space, self.W_out.unsqueeze(0)) + self.b_out

        return out

    @staticmethod
    def test(
        sentence: str, tokenizer: PreTrainedTokenizerBase, reference_gpt2: HookedTransformer
    ) -> None:
        if tokenizer is not None:
            logits, cache = reference_gpt2.run_with_cache(sentence)
            Tests.load_gpt2_test(
                MLP, cast("nn.Module", reference_gpt2.blocks[0].mlp), cache["normalized", 0, "ln2"]
            )


class TransformerBlock(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.ln1 = LayerNorm(cfg)
        self.attn = Attention(cfg)
        self.ln2 = LayerNorm(cfg)
        self.mlp = MLP(cfg)

    def forward(
        self, resid_pre: Float[Tensor, "batch position d_model"]
    ) -> Float[Tensor, "batch position d_model"]:
        # Attention
        ln_1_out = self.ln1(resid_pre)
        attention_out = self.attn(ln_1_out)
        # And adding this to the residual
        residual_post_attention = resid_pre + attention_out
        # MLP
        ln_2_out = self.ln2(residual_post_attention)
        mlp_out = self.mlp(ln_2_out)
        # And adding this to the residual
        residual_post_mlp: Float[Tensor, "batch position d_model"] = (
            residual_post_attention + mlp_out
        )

        return residual_post_mlp

    @staticmethod
    def test(
        sentence: str, tokenizer: PreTrainedTokenizerBase, reference_gpt2: HookedTransformer
    ) -> None:
        if tokenizer is not None:
            logits, cache = reference_gpt2.run_with_cache(sentence)
            Tests.load_gpt2_test(TransformerBlock, reference_gpt2.blocks[0], cache["resid_pre", 0])

    @staticmethod
    def test_with_random() -> None:
        Tests.rand_float_test(TransformerBlock, [2, 4, 768])


class Unembed(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.W_U = nn.Parameter(t.empty((cfg.d_model, cfg.d_vocab)), requires_grad=True)
        nn.init.normal_(self.W_U, std=self.cfg.init_range)
        self.b_U = nn.Parameter(t.zeros((cfg.d_vocab)), requires_grad=True)

    def forward(
        self, normalized_resid_final: Float[Tensor, "batch position d_model"]
    ) -> Float[Tensor, "batch position d_vocab"]:
        out= t.matmul(normalized_resid_final, self.W_U) + self.b_U
        return out

    @staticmethod
    def test(
        sentence: str, tokenizer: PreTrainedTokenizerBase, reference_gpt2: HookedTransformer
    ) -> None:
        if tokenizer is not None:
            logits, cache = reference_gpt2.run_with_cache(sentence)
            Tests.load_gpt2_test(Unembed, reference_gpt2.unembed, cache["ln_final.hook_normalized"])

    @staticmethod
    def test_with_random() -> None:
        Tests.rand_float_test(Unembed, [2, 4, 768])


class DemoTransformer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.embed = Embed(cfg)
        self.pos_embed = PosEmbed(cfg)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_final = LayerNorm(cfg)
        self.unembed = Unembed(cfg)
        self.reference = HookedTransformer.from_pretrained(
            "gpt2-small",
            fold_ln=False,
            center_unembed=False,
            center_writing_weights=False,  # you'll learn about these arguments later!
        )
        self.cfg.d_vocab = self.reference.cfg.d_vocab
        assert self.reference.tokenizer is not None, "Reference does not have a tokenizer."
        self.tokenizer: PreTrainedTokenizerBase = self.reference.tokenizer

    @jaxtyped(typechecker=typechecker)
    def forward(
        self, tokens: Int[Tensor, "batch position"] | list[int]
    ) -> Float[Tensor, "batch position d_vocab"]:
        # Ensure the tokens are already a tensor. if not, convert:
        if not isinstance(tokens, Tensor):
            tokens = Tensor(tokens).to(t.int).to(device)
        # Embed meaning and position
        x_embed = self.embed(tokens)
        x_pos = self.pos_embed(tokens)
        x_0 = x_embed + x_pos
        # Transform through all the blocks, keep the intermediate outputs
        x_res: list[Float[Tensor, "batch position d_model"]] = [x_0]
        for transformer_block in self.blocks:
            x_res.append(
                transformer_block(x_res[-1])
            )  # take the latest residual stream and run it through this block
        logits: Float[Tensor, "batch position d_vocab"] = self.unembed(self.ln_final(x_res[-1]))
        return logits

    def complete_text(self, sentence: str, length: int = 100) -> str:
        out_sentence = sentence
        for _ in tqdm(range(length)):
            test_tokens = (
                Tensor(self.tokenizer.encode(out_sentence)).to(t.int).to(device).unsqueeze(0)
            )
            demo_logits = self(test_tokens)
            next_token = demo_logits[-1, -1].argmax()
            decoded = self.tokenizer.decode(next_token)
            # Added as the decoder seems to either return a single str or a list[str]
            if isinstance(decoded, list):
                decoded = decoded[0] if decoded else ""
            out_sentence += decoded
        return out_sentence

    def load_pretrained_weights_from_reference(self) -> None:
        self.load_state_dict(self.reference.state_dict(), strict=False)
        return

    @staticmethod
    def test(
        sentence: str, tokenizer: PreTrainedTokenizerBase, reference_gpt2: HookedTransformer
    ) -> None:
        if tokenizer is not None:
            tokens = Tensor(tokenizer.encode(sentence)).to(device).to(t.int)
            Tests.load_gpt2_test(DemoTransformer, reference_gpt2, tokens)

    @staticmethod
    def test_with_random() -> None:
        Tests.rand_int_test(DemoTransformer, [2, 4])

class TransformerSampler:
    def __init__(self, model: DemoTransformer, tokenizer: GPT2TokenizerFast):
        self.model = model
        self.cfg = model.cfg
        self.tokenizer = tokenizer

    @t.inference_mode()
    def sample(self, prompt: str, max_tokens_generated=100, verbose=False, **kwargs) -> str:
        """
        Returns a string of autoregressively generated text, starting from the prompt.

        Sampling terminates at max_tokens_generated, or when the model generates an end-of-sequence token. kwargs are
        passed to sample_next_token, to give detailed instructions on how new tokens are chosen. 
        Pass `seed` to make generation reproducible.
        """
        sequence: list[str] = []
        self.model.eval()
        seed = kwargs.pop("seed", None)
        if seed is not None:
            t.manual_seed(seed)
            np.random.seed(seed)
        while len(sequence) <= max_tokens_generated and sequence[-1] != self.tokenizer.eos_token_id:
            prompt_with_new_seq = prompt + "".join(sequence)
            input_tokens = Tensor(self.tokenizer.encode(prompt_with_new_seq)).to(t.int).to(device).unsqueeze(0)
            logits:  Float[Tensor, "batch position d_vocab"] = self.model.forward(input_tokens) # For sampling, shouldn't we just keep the last position ? I will just keep the last position for now:
            last_logits: Float[Tensor, "batch d_vocab"] = logits[:, -1, :]
            next_token = self.sample_next_token(input_tokens.squeeze(0), last_logits.squeeze(0))
            sequence.append(str(self.tokenizer.decode(next_token))) # Add the new word to the sequence
            if verbose:
                print(prompt + "".join(sequence), end="\r")
        return prompt + "".join(sequence)


    @staticmethod
    def sample_next_token(
        input_ids: Int[Tensor, " seq_len"],
        logits: Float[Tensor, "d_vocab"],
        temperature=1.0,
        top_k=0,
        top_p=0.0,
        frequency_penalty=0.0,
    ) -> int:
        assert input_ids.ndim == 1, "input_ids should be a 1D sequence of token ids"
        assert logits.ndim == 1, "logits should be a 1D tensor of shape (d_vocab,)"
        assert temperature >= 0, "Temperature should be non-negative"
        assert 0 <= top_p <= 1.0, "Top-p must be a probability"
        assert 0 <= top_k, "Top-k must be non-negative"
        assert not (top_p != 0 and top_k != 0), "At most one of top-p and top-k supported"

        # Apply all the specialized sampling methods
        if temperature == 0:
            return TransformerSampler.greedy_search(logits)
        elif temperature != 1.0:
            logits = TransformerSampler.apply_temperature(logits, temperature)
        if frequency_penalty != 0.0:
            logits = TransformerSampler.apply_frequency_penalty(input_ids, logits, frequency_penalty)
        if top_k > 0:
            return TransformerSampler.sample_top_k(logits, top_k)
        if top_p > 0.0:
            return TransformerSampler.sample_top_p(logits, top_p)
        return TransformerSampler.sample_basic(logits)

    @staticmethod
    def greedy_search(logits: Float[Tensor, "d_vocab"]) -> int:
        """
        Returns the most likely token (as an int).
        """
        raise NotImplementedError()

    @staticmethod
    def apply_temperature(logits: Float[Tensor, "d_vocab"], temperature: float) -> Float[Tensor, "d_vocab"]:
        """
        Applies temperature scaling to the logits.
        """
        raise NotImplementedError()

    @staticmethod
    def apply_frequency_penalty(
        input_ids: Int[Tensor, " seq_len"], logits: Float[Tensor, "d_vocab"], freq_penalty: float
    ) -> Float[Tensor, "d_vocab"]:
        """
        Applies a frequency penalty to the logits.
        """
        raise NotImplementedError()

    @staticmethod
    def sample_basic(logits: Float[Tensor, "d_vocab"]) -> int:
        """
        Samples from the distribution defined by the logits.
        """
        raise NotImplementedError()

    @staticmethod
    def sample_top_k(logits: Float[Tensor, "d_vocab"], k: int) -> int:
        """
        Samples from the top k most likely tokens.
        """
        raise NotImplementedError()

    @staticmethod
    def sample_top_p(logits: Float[Tensor, "d_vocab"], top_p: float, min_tokens_to_keep: int = 1) -> int:
        """
        Samples from the most likely tokens which make up at least p cumulative probability.
        """
        raise NotImplementedError()

    @t.inference_mode()
    def beam_search(
        self,
        prompt: str,
        num_return_sequences: int,
        num_beams: int,
        max_new_tokens: int,
        no_repeat_ngram_size: int | None = None,
    ) -> list[tuple[float, str]]:
        """
        Implements a beam search, by repeatedly performing the `generate` and `filter` steps (starting from the initial
        prompt) until either of the two stopping criteria are met: (1) we've generated `max_new_tokens` tokens, or (2)
        we've generated `num_returns_sequences` terminating sequences.
        """
        raise NotImplementedError()
    def test_greedy(self) -> None:
        expected = "Jingle bells, jingle bells, jingle all the way up to the top of the mountain."
        prompt = "Jingle bells, jingle bells, jingle all the way"
        print(f"Testing greedy decoding\nPrompt:   {prompt!r}")
        output = self.sample(prompt, max_tokens_generated=8, temperature=0.0)
        print(f"Expected: {expected!r}\nActual:   {output!r}\n")
        assert output == expected
        print("Tests passed!")

