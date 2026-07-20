# MLP-Namegen-w-Batchnorm

# MLP Name Generator with BatchNorm

A character-level neural network that learns to generate names using a multilayer perceptron (MLP) with custom batch normalization layers.

This project is inspired by Andrej Karpathy’s *makemore* series and implements a small neural language model from scratch using PyTorch tensors and manual layer classes.

## Overview

The model learns patterns from a dataset of names and generates new name-like strings one character at a time.

Each training example uses a fixed-length context of previous characters to predict the next character. The model embeds characters into vectors, flattens the context, passes it through several MLP layers, and outputs logits over the vocabulary.

## Features

- Character-level name generation
- Learned character embeddings
- 4-character context window
- Train / validation / test split
- Custom neural network layers:
  - `Linear`
  - `Tanh`
  - `BatchNorm1d`
- Batch normalization with running mean and variance
- Cross-entropy training objective
- Sampling loop for generating new names

## Project Structure

```text
MLP-Namegen-w-Batchnorm/
├── README.md
├── main.py
├── nn.py
├── batchnorm.py
└── names.txt
```

### `main.py`

Main training script. It:

- Loads the names dataset
- Builds the character vocabulary
- Creates train, validation, and test splits
- Defines the model architecture
- Trains the model with mini-batch gradient descent
- Evaluates loss on each split
- Samples new names from the trained model

### `nn.py`

Contains custom implementations of the model layers:

- `Linear`: affine transformation layer
- `Tanh`: activation layer
- `BatchNorm1d`: batch normalization layer with learnable scale and shift parameters

### `names.txt`

Dataset of names used for training.

## Model Architecture

The model uses:

- Vocabulary size: `27`
- Embedding size: `10`
- Context size: `4`
- Hidden size: `300`
- Batch size: `64`

The architecture is a deep MLP:

```text
Embedding
→ Flatten context
→ Linear
→ BatchNorm1d
→ Tanh
→ Linear
→ BatchNorm1d
→ Tanh
→ Linear
→ BatchNorm1d
→ Tanh
→ Linear
→ BatchNorm1d
→ Tanh
→ Linear
→ BatchNorm1d
→ Tanh
→ Linear
→ BatchNorm1d
→ Character logits
```

The final logits are trained with cross-entropy loss to predict the next character.

## How It Works

The model treats name generation as a next-character prediction problem.

For example, with a context size of 4, the model sees four previous characters and predicts the next one:

```text
.... → e
...e → m
..em → m
.emm → a
emma → .
```

The period `.` is used as a special start/end token.

During generation, the model begins with an empty context and repeatedly samples the next character until it produces the end token.

## Installation

Clone the repository:

```bash
git clone https://github.com/Benswe/MLP-Namegen-w-Batchnorm.git
cd MLP-Namegen-w-Batchnorm
```

Install dependencies:

```bash
pip install torch matplotlib
```

## Usage

Run the training script:

```bash
python main.py
```

The script will train the model, print the train/validation/test losses, and then sample generated names.

## Example Output

After training, the model will print generated names like:

```text
mariel.
kaeli.
avanna.
jore.
```

Actual outputs will vary because sampling is random.

## Learning Goals

This project is mainly educational. It is designed to build intuition for:

- Character-level language modeling
- Embeddings
- MLP architectures
- Batch normalization
- Training/validation/test splits
- Cross-entropy loss
- Sampling from a neural network
- Building neural network components from scratch

## Notes

This project intentionally avoids using high-level PyTorch modules like `torch.nn.Linear` or `torch.nn.BatchNorm1d`. Instead, the core layers are implemented manually to better understand how neural networks work under the hood.

## Future Improvements

Possible extensions:

- Add a cleaner training loop
- Create an experimental harness to improve hyperparameters
- Use a WaveNet-style architecture to preserve local character structure instead of flattening the entire context into the first layer.
