from collections import defaultdict

import torch as t
from torch import Tensor
from tqdm import tqdm
from transformer_lens import HookedTransformer

import tests
from custom_transformer import Config, DemoTransformer, TransformerSampler
from training import TransformerTrainer, TransformerTrainingArgs

device = t.device(
    "mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu"
)
# Training model config
""" model_cfg = Config(
    debug=False,
    d_model=32,
    n_heads=16,
    d_head=2,
    d_mlp=32 * 4,
    n_layers=4,
    n_ctx=128,
    # d_vocab will be taken from the ref model automatically
) """
model = DemoTransformer(Config()).to(device)
model.load_pretrained_weights_from_reference()
sampler = TransformerSampler(model, model.tokenizer) # type: ignore

tests.test_apply_frequency_penalty(TransformerSampler.apply_frequency_penalty)

bieber_prompt = "And I was like Baby, baby, baby, oh Like, Baby, baby, baby, no Like, Baby, baby, baby, oh I thought you'd always be mine, mine"
input_ids = Tensor(model.tokenizer.encode(bieber_prompt, return_tensors="pt")).to(device)
logits = t.ones(model.tokenizer.vocab_size)
penalized_logits = TransformerSampler.apply_frequency_penalty(input_ids.squeeze(), logits, 2.0)

assert penalized_logits[5156].item() == -11, "Expected 6 occurrences of ' baby' with leading space, 1-2*6=-11"
assert penalized_logits[14801].item() == -5, "Expected 3 occurrences of ' Baby' with leading space, 1-2*3=-5"

print("Tests passed!")

