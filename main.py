import torch as t

from custom_transformer import Config, DemoTransformer, TransformerSampler
from training import TransformerTrainer, TransformerTrainingArgs

device = t.device(
    "mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu"
)
model_cfg = Config(
    debug=False,
    d_model=32,
    n_heads=16,
    d_head=2,
    d_mlp=32 * 4,
    n_layers=4,
    n_ctx=128,
    # d_vocab will be taken from the ref model automatically
)
model = DemoTransformer(model_cfg).to(device)
sampler = TransformerSampler(model, model.tokenizer) # type: ignore
sampler.test_greedy()


