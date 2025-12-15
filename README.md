# MiniGPT - Shakespeare Model

## Project Description
This project consists of a from-scratch implementation of a Decoder-Only Transformer architecture (GPT-style), trained to generate dramatic text in the style of William Shakespeare.

Unlike projects using pre-built libraries like Hugging Face, this model was built layer-by-layer using native PyTorch to fundamentally understand the exact gradient flow and the mathematics behind:

Self-Attention Mechanism (Scaled Dot-Product Attention).

Vectorized Multi-Head Attention.

Sinusoidal Positional Encodings (Manual mathematical implementation, not learned embeddings).

The model was trained on the "Tiny Shakespeare" dataset and achieves a competitive validation Loss (~1.48).

## Key Features
Manual Architecture: Custom implementation of Transformer blocks, LayerNorm (Pre-Norm), and Causal Masking.

Decoding Strategies: Support for three text generation methods:

Multinomial Sampling: For creative and varied results.

Beam Search: For coherent and structured results.

Greedy Decoding: For deterministic decoding.

Visualization: Automatic generation of Heatmaps for Positional Encodings and  Loss curves.

Tokenizer: Character-level tokenizer.

## Requirements
To run this code, you need to have the following installed:

Python 3

PyTorch

Matplotlib (for plotting)

Numpy

Requests (to download the dataset)

You can install the dependencies with:

Bash
pip install torch matplotlib numpy requests
## Installation & Usage
Clone the repository:

git clone https://github.com/arojocz/mini-gpt.git

Train the model: Run the main script. If no saved model exists, training will start automatically (3000 iterations by default).

Bash
python minigpt.py
Interactive Mode: Upon completion of training, the script enters chat mode. You can type a "prompt" and watch the model complete it using all three decoding strategies simultaneously.

## Architecture & Results
The model uses the following hyperparameter configuration:

Embedding Dimension: 384

Context Length (Block Size): 128

Heads: 6

block: 4

Parameters: ~7.14 Million

## Example

begin:  > to be or not to  

--------------- [Multinomial] ---------------
to be or not to do saw myself,
And with love, from my joy cakes O, tell me,
The rest of my thus name fearful that ca

--------------- [Beam Search] ---------------
to be or not to the father.

KING RICHARD III:
My lord, and my lord, and my lord, and my lord.

QUEEN ELIZABETH:
The

--------------- [Greedy] ---------------
to be or not to the soul of the son,
And the sea of the sea of the state, and the state
That was the seased of the


## Additional Documentation

* [Descargar Reporte (PDF)](./docs/mini_GPT_project2.pdf)