
from dataclasses import dataclass
from typing import Tuple, List
from beartype import beartype as typechecker
import torch as t
import torch.nn as nn
from jaxtyping import Float, Int, jaxtyped
from torch import Tensor
from transformer_lens import HookedTransformer

from visualizers import display_logits

from transformer_lens.utils import gelu_new
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
    def rand_float_test(model_cls :  type[nn.Module] , shape: list[int], **args) -> None:
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
        one_hot_tokens : Int[Tensor, "batch position d_vocab"] = nn.functional.one_hot(tokens.to(t.long), self.cfg.d_vocab)#noqa F722
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
        pos_embed = t.matmul(position_tensor, self.W_pos)[:, :tokens.shape[-1], :]
        return pos_embed
    
    @staticmethod
    def test(sentence: str):
        if tokenizer is not None: # Only did this to satisfy my Linter
            Tests.load_gpt2_test(PosEmbed, reference_gpt2.pos_embed, t.tensor(tokenizer.encode(sentence)).to(device))
    
    @staticmethod
    def test_with_random():
        Tests.rand_int_test(PosEmbed, [2, 4])

class Attention(nn.Module):
    IGNORE: Float[Tensor, ""]

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_Q = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_K = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_V = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_O = nn.Parameter(t.empty((cfg.n_heads, cfg.d_head, cfg.d_model)))
        self.b_Q = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.b_K = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.b_V = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.b_O = nn.Parameter(t.zeros((cfg.d_model)))
        nn.init.normal_(self.W_Q, std=self.cfg.init_range)
        nn.init.normal_(self.W_K, std=self.cfg.init_range)
        nn.init.normal_(self.W_V, std=self.cfg.init_range)
        nn.init.normal_(self.W_O, std=self.cfg.init_range)
        self.register_buffer("IGNORE", t.tensor(float("-inf"), dtype=t.float32, device=device))
    
    @jaxtyped(typechecker=typechecker)
    def forward(self, normalized_resid_pre: Float[Tensor, "batch posn d_model"]) -> Float[Tensor, "batch posn d_model"]:
        # 1. Compute query key vectors
        Q : Float[Tensor, "batch n_heads posn d  d_head"] = t.matmul(normalized_resid_pre.unsqueeze(1), self.W_Q.unsqueeze(0)) + self.b_Q.unsqueeze(0).unsqueeze(2)
        K : Float[Tensor, "batch n_heads posn d  d_head"] = t.matmul(normalized_resid_pre.unsqueeze(1), self.W_K.unsqueeze(0)) + self.b_K.unsqueeze(0).unsqueeze(2)
        # 2. In parallel, compute the value vectors
        V : Float[Tensor, "batch n_heads posn d  d_head"] = t.matmul(normalized_resid_pre.unsqueeze(1), self.W_V.unsqueeze(0)) + self.b_V.unsqueeze(0).unsqueeze(2)
        # 3. Create the full attention pattern (no masking, softmax yet)
        QK : Float[Tensor, "batch n_heads posn d posn_d"]= t.matmul(Q, K.transpose(2, 3))
        # 4. Scale the attention matrix to avoid vanishing gradients
        QK = QK/t.sqrt(Tensor([self.cfg.d_head]).to(device).to(t.float))
        # 5. Mask key indexes higher than query indexes to force the model to look back only..
        QK_masked : Float[Tensor, "batch n_heads posn d posn_d"] = self.apply_causal_mask(QK)
        # 6. Convert into a probability distribution for each query row, along the key column dimension:
        QK_p : Float[Tensor, "batch n_heads posn d posn_d"] = t.softmax(QK_masked, 3)
        # 7. Do the weighted average of the value vectors using the key vectors
        V_avg : Float[Tensor, "batch n_heads posn d d_head"] = t.matmul(QK_p, V)
        # 8. Linear layer with scale, sum the heads and add bias to finish before the output of the block:
        O : Float[Tensor, "batch posn d d_model"] = t.matmul(V_avg, self.W_O).sum(dim=1)+self.b_O
        
        # ave_image(QK_p[0,0,:,:], 'GREY_img.png')

        return O

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
    def test(sentence : str):
        if tokenizer is not None:
            logits, cache = reference_gpt2.run_with_cache(sentence)
            display(
                cv.attention.attention_patterns(
                    tokens=reference_gpt2.to_str_tokens(sentence), attention=cache["pattern", 0][0]
                )
            )
            Tests.load_gpt2_test(Attention, reference_gpt2.blocks[0].attn, cache["normalized", 0, "ln1"]) 
    
    def test_causal_mask(self):
        input = t.rand([2, self.cfg.d_head, self.cfg.d_model // self.cfg.d_head, self.cfg.d_model // self.cfg.d_head]).to(device)        
        print("Input shape:", input.shape)
        output = self.apply_causal_mask(input)
        if isinstance(output, tuple):
            output = output[0]
        print("Output shape:", output.shape, "\n")
        print(output[0,0,:,:])

class MLP(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_in = nn.Parameter(t.empty((cfg.d_model, cfg.d_mlp)))
        self.W_out = nn.Parameter(t.empty((cfg.d_mlp, cfg.d_model)))
        self.b_in = nn.Parameter(t.zeros((cfg.d_mlp)))
        self.b_out = nn.Parameter(t.zeros((cfg.d_model)))
        nn.init.normal_(self.W_in, std=self.cfg.init_range)
        nn.init.normal_(self.W_out, std=self.cfg.init_range)

    def forward(self, normalized_resid_mid: Float[Tensor, "batch posn d_model"]) -> Float[Tensor, "batch posn d_model"]:
        
        hidden_space =  gelu_new(t.matmul(normalized_resid_mid, self.W_in.unsqueeze(0)) + self.b_in)
        out = t.matmul(hidden_space, self.W_out.unsqueeze(0)) + self.b_out

        return out
        
    
    @staticmethod
    def test(sentence : str):
        if tokenizer is not None:
            logits, cache = reference_gpt2.run_with_cache(sentence)
            Tests.load_gpt2_test(MLP, reference_gpt2.blocks[0].mlp, cache["normalized", 0, "ln2"])

class TransformerBlock(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.ln1 = LayerNorm(cfg)
        self.attn = Attention(cfg)
        self.ln2 = LayerNorm(cfg)
        self.mlp = MLP(cfg)

    def forward(self, resid_pre: Float[Tensor, "batch position d_model"]) -> Float[Tensor, "batch position d_model"]:
        # Attention
        ln_1_out = self.ln1(resid_pre)
        attention_out = self.attn(ln_1_out)
        # And adding this to the residual
        residual_post_attention = resid_pre + attention_out
        # MLP
        ln_2_out = self.ln2(residual_post_attention)
        mlp_out = self.mlp(ln_2_out)
        # And adding this to the residual
        residual_post_mlp = residual_post_attention + mlp_out

        return residual_post_mlp
    
    @staticmethod
    def test(sentence):
        if tokenizer is not None:
            logits, cache = reference_gpt2.run_with_cache(sentence)
            Tests.load_gpt2_test(TransformerBlock, reference_gpt2.blocks[0], cache["resid_pre", 0])
    
    @staticmethod
    def test_with_random():
        Tests.rand_float_test(TransformerBlock, [2, 4, 768])

class Unembed(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_U = nn.Parameter(t.empty((cfg.d_model, cfg.d_vocab)))
        nn.init.normal_(self.W_U, std=self.cfg.init_range)
        self.b_U = nn.Parameter(t.zeros((cfg.d_vocab), requires_grad=False))

    def forward(
        self, normalized_resid_final: Float[Tensor, "batch position d_model"]
    ) -> Float[Tensor, "batch position d_vocab"]:
        
        return t.matmul(normalized_resid_final, self.W_U) + self.b_U
    
    @staticmethod
    def test(sentence):
        if tokenizer is not None:
            logits, cache = reference_gpt2.run_with_cache(sentence)
            Tests.load_gpt2_test(Unembed, reference_gpt2.unembed, cache["ln_final.hook_normalized"])
    
    @staticmethod
    def test_with_random():
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

    def forward(self, tokens: Int[Tensor, "batch position"] | list) -> Float[Tensor, "batch position d_vocab"]:
        # Ensure the tokens are already a tensor. if not, convert:
        if type(tokens) is not Int[Tensor, '...']:
            tokens = Tensor(tokens).to(t.int).to(device)
        # Embed meaning and position
        x_embed = self.embed(tokens)
        x_pos = self.pos_embed(tokens)
        x_0 = x_embed + x_pos
        # Transform through all the blocks, keep the intermediate outputs        
        x_res : List[Float[Tensor, "batch position d_model"]] = [x_0] 
        for transformer_block in self.blocks:
            x_res.append(transformer_block(x_res[-1])) # take the latest residual stream and run it through this block
        logits = self.unembed(self.ln_final(x_res[-1]))
        return logits
    @staticmethod
    def test(sentence):
        if tokenizer is not None:
            tokens = Tensor(tokenizer.encode(sentence)).to(device).to(t.int)
            Tests.load_gpt2_test(DemoTransformer, reference_gpt2, tokens)
    
    @staticmethod
    def test_with_random():
        Tests.rand_int_test(DemoTransformer, [2, 4])

if __name__ == "__main__":
    cache = None
    if tokenizer is None:
        raise TypeError
    sentence = "When will the earth stop moving ? Or when will the pigs fly ? The answer does not lie in the  "
    demo_gpt2 = DemoTransformer(Config(debug=False)).to(device)
    demo_gpt2.load_state_dict(reference_gpt2.state_dict(), strict=False)
    tokens = tokenizer.encode(sentence)
    demo_logits = demo_gpt2(tokens)
    display_logits(tokenizer, tokens, demo_logits)
