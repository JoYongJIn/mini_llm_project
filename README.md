# Mini LLM Training Pipeline

A hands-on implementation of a small language model training pipeline,
covering tokenization, model components, training, and fine-tuning.

This project was developed to understand the internal components of
large language models by implementing and experimenting with each stage
of the pipeline directly in Python and PyTorch.

## Project Overview

Instead of treating an LLM as a black box, I explored the components
required to build and train a language model.

The project includes experiments with tokenization, BPE,
neural-network architectures, model training, and fine-tuning.

## Pipeline

```text
Raw Text
   ↓
Tokenizer / BPE
   ↓
Tokenized Dataset
   ↓
Language Model
   ↓
Training
   ↓
Evaluation
   ↓
Fine-tuning
