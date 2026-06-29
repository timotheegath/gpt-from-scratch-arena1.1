# Arena 1.1 – Clean Transformer implementation

This repository is my personal implementation of the clean Transformer from Arena Chapter 1.1, "Transformer Interpretability", closely following the tutorial here:

<https://learn.arena.education/chapter1_transformer_interp/01_transformers/2-clean-transformer-implementation/>

The aim is to understand and re‑implement the core components of a GPT‑2–style Transformer, with simple tests around each layer to check that the implementation behaves as expected.

Personally, this was my first time implementing a transformer end-to-end, and I had plenty of fun and "aha!" moments while implementing it. My last implementation of a model architecture, evaluation and training loop goes back to 2017, where [I trained and evaluated a Deep Neural Q-Network to learn a policy to play Atari games](https://github.com/timotheegath/DeepNetworkAtariRL)

## Disclaimers
- In some parts, I have replaced my code attempt with the solution from the ARENA guide. This was usually because the implementation from ARENA was clearer than mine and would allow me to refer back to it easily as I reuse this code. 
- In the case of beam search, I got stuck, commented my own code, explained my mistake, and replaced it with the ARENA solution
- I split this into multiple files, contrary to the ARENA implementation, to make it more reusable in the future

## Getting started

This project uses `uv` for Python environment and dependency management.

1. Install `uv` (see the `uv` docs for your platform).
2. From the repo root, create/sync the environment:
   - `uv sync`
3. Run the main script:
   - `uv run python main.py`


## Code structure

- `custom_transformer.py`: core implementation of the custom Transformer building blocks (LayerNorm, embedding layers, etc.)
- `main.py`: entry point for running experiments or quick demos with the model
- `training.py`: contains the `TransformerTrainer` class and `TransformerTrainingArgs` for training the model and logging metrics using Weights & Biases (wandb)

## Main Functions and Classes in `custom_transformer.py`

The `custom_transformer.py` script contains the core implementation of the custom Transformer building blocks. Key classes include:

- **DemoTransformer**: A custom implementation of a GPT-2-style Transformer.
- **TransformerSampler**: A utility class for sampling text from the `DemoTransformer`.
- **LayerNorm**: Implements layer normalization.
- **Embed**: Handles token embeddings.
- **PosEmbed**: Manages positional embeddings.
- **Attention**: Implements the attention mechanism.
- **MLP**: Multi-layer perceptron for feed-forward processing.
- **TransformerBlock**: Combines attention and MLP layers into a single block.
- **Unembed**: Converts model outputs back to token logits.

## Using the Model in `main.py`

To use the model in `main.py`, follow these steps:

1. **Declare the CUDA device**:
   ```python
   device = t.device(
       "mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu"
   )
   ```

2. **Create the configuration**:
   Either use the default GPT-2 structure by just using `Config()` or specify custom parameters:
   ```python
   model_cfg = Config(
       debug=False,
       d_model=32,
       n_heads=16,
       d_head=2,
       d_mlp=32 * 4,
       n_layers=4,
       n_ctx=128,
   )
   ```

3. **Create the transformer object**:
   ```python
   model = DemoTransformer(model_cfg).to(device)
   ```

4. **Load pretrained weights (optional)**:
   ```python
   model.load_pretrained_weights_from_reference()
   ```
   Note that the reference is set as gpt-2 from HookedTransformer
   ```python
   HookedTransformer.from_pretrained(
            "gpt2",
            fold_ln=False,
            center_unembed=False,
            center_writing_weights=False,  # you'll learn about these arguments later!
        )
   ```

5. **Use the model**:
   ```python
   sampler = TransformerSampler(model, model.tokenizer)
   your_prompt = "Your prompt here..."
   completions = sampler.beam_search(your_prompt, num_return_sequences=3, num_beams=4, max_new_tokens=40)
   ```

## Training the Model

The `training.py` script provides the `TransformerTrainer` class for training the model and logging metrics using Weights & Biases (wandb). Key components include:

- **TransformerTrainer**: Manages the training process, including:
  - `training_step`: Performs a single training step, calculates loss, and logs it to wandb.
  - `evaluate`: Evaluates the model on the test set and logs accuracy to wandb.
  - `train`: Runs the training loop for the specified number of epochs and logs metrics to wandb.

- **TransformerTrainingArgs**: Configuration for training, including:
  - `batch_size`: Batch size for training.
  - `epochs`: Number of training epochs.
  - `max_steps_per_epoch`: Maximum number of steps per epoch.
  - `lr`: Learning rate.
  - `weight_decay`: Weight decay for the optimizer.
  - `wandb_project`: Name of the wandb project for logging.
  - `wandb_name`: Name of the wandb run.

### Running Training with wandb Logging

To run training with wandb logging:

1. Ensure you have wandb installed and configured:
   - `uv add wandb`
   - `wandb login`

2. Import the necessary classes into `main.py`:
   ```python
   from training import TransformerTrainer, TransformerTrainingArgs
   ```

3. Configure the training arguments:
   ```python
   training_args = TransformerTrainingArgs(
       batch_size=32,
       epochs=10,
       max_steps_per_epoch=500,
       lr=1e-3,
       weight_decay=1e-2,
       wandb_project="day1-demotransformer-for-real",
       wandb_name="my-training-run"
   )
   ```

4. Initialize the trainer and start training:
   ```python
   trainer = TransformerTrainer(training_args, model)
   trainer.train()
   ```

## Layer-level test functions

Each custom layer in `custom_transformer.py` is written to be easy to test in isolation.
Individual layers define small static test helpers that call into `Tests`, for example:
- `LayerNorm.test(sentence: str)` runs the custom `LayerNorm` on cached GPT‑2 activations and checks that the outputs match GPT‑2’s final layer norm
- `Embed.test(sentence: str)` checks that the custom embedding layer matches GPT‑2’s embedding on a tokenized sentence

The idea is that you can iteratively implement or modify a layer, then quickly run its test function to confirm that:

- Input and output shapes look sensible
- The layer does not modify inputs in place
- The output is numerically close to GPT‑2’s behaviour where applicable

These tests keep the feedback loop tight while working through the Arena chapter.

## Tooling – `uv` and `ruff`

The tooling is intentionally minimal, the goal was not produe the cleanest code, but still keep it reusable.

- `uv` manages the Python environment and dependencies, keeping setup fast and reproducible.
- `ruff` is used as a linter (and optionally formatter) to maintain a consistent, clean code style while iterating on the implementation. Typical commands are:
  - `ruff check .`
  - `ruff check . --fix`
