from custom_transformer import Config, DemoTransformer
from training import TransformerTrainingArgs, load_dataset

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
model = DemoTransformer(model_cfg)
training_params = TransformerTrainingArgs()
assert model.reference.tokenizer is not None
train_loader, test_loader = load_dataset(
    model.reference.tokenizer, model.cfg.n_ctx, training_params
)
print()
