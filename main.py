from collections import defaultdict

import torch as t
from torch import Tensor
from tqdm import tqdm
from transformer_lens import HookedTransformer

import tests
import wandb
from custom_transformer import Config, DemoTransformer, TransformerSampler
from training import TransformerTrainer, TransformerTrainingArgs
from wandb import Table

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
sampler = TransformerSampler(model, model.tokenizer)  # type: ignore

your_prompt = "In a shocking finding, scientist discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains. Even more surprising to the researchers was the fact that the unicorns spoke perfect English."
orig_len = len(sampler.tokenizer.encode(your_prompt))
final_logitsums_and_completions = sampler.beam_search(your_prompt, 3, 40, 60, 2)
# Print all the best output
for logprob_sum, text in final_logitsums_and_completions:
    avg_logprob_as_prob = t.tensor(logprob_sum / (len(sampler.tokenizer.encode(text)) - orig_len)).exp()
    print(f"Avg token prob = {avg_logprob_as_prob:.3f}\nBest output:\n[bold dark_orange]{text}")

