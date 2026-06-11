from collections import defaultdict

import torch as t
from torch import Tensor
from tqdm import tqdm

from custom_transformer import Config, DemoTransformer, TransformerSampler
from transformer_lens import HookedTransformer
from training import TransformerTrainer, TransformerTrainingArgs
from tests import test_sample_basic

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
test_sample_basic(sampler.sample_basic)
prompt = "John and Mary went to the"
input_ids = Tensor(model.tokenizer.encode(prompt, return_tensors="pt")).to(device)
logits = model(input_ids)[0, -1]

expected_top_5 = {
    " church": 0.0648,
    " house": 0.0367,
    " temple": 0.0145,
    " same": 0.0104,
    " Church": 0.0097,
}
frequency_of_top_5: defaultdict[str, int] = defaultdict(int)

N = 10_0000
for _ in tqdm(range(N)):
    token = TransformerSampler.sample_next_token(input_ids.squeeze(), logits)
    frequency_of_top_5[str(model.tokenizer.decode(token))] += 1

for word in expected_top_5:
    expected_freq = expected_top_5[word]
    observed_freq = frequency_of_top_5[word] / N
    print(f"Word: {word!r:<9}. Expected freq {expected_freq:.4f}, observed freq {observed_freq:.4f}")
    assert abs(observed_freq - expected_freq) < 0.01, "Try increasing N if this fails by a small amount."

print("Tests passed!")


