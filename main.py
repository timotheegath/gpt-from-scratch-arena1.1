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

N_RUNS = 2
your_prompt = "Jingle bells, jingle bells, jingle all the way"
cases = [
    ("High freq penalty", dict(frequency_penalty=100.0)),
    ("Negative freq penalty", dict(frequency_penalty=-3.0)),
    ("Too hot!", dict(temperature=2.0)),
    ("Pleasantly cool", dict(temperature=0.7)),
    ("Pleasantly warm", dict(temperature=0.9)),
    ("Too cold!", dict(temperature=0.01)),
]

table = Table(columns=["Name", "Kwargs", "Output"])
with wandb.init(project="temperature-penalty-test") as run:
    for name, kwargs in cases:
        for i in range(N_RUNS):
            output = sampler.sample(your_prompt, max_tokens_generated=24)
            table.add_row(name, str(kwargs), repr(output) + "\n")
    run.log({"Sampling - Manual Testing": table})
    print(table)
