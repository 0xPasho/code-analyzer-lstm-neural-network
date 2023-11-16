# codesapiens.ai Neural Network (Proof of concept)

This is a seq2seq neural network based in [Code5T paper](https://arxiv.org/abs/2109.00859). This repo is ONLY focused on generating analyzing code and giving a summary from it.

## Try it yourself

Try it on Hugging Face with [this link](https://huggingface.co/pasho/codesapiens-poc-code-summarization).

## How to train it ?

You should be good with the $300 dlls given from Google, spin off a VM in GCloud with 32GB of RAM, but don't forget to turn if off...

## Scripts Overview

train_model.py: The main script that orchestrates training, evaluation, and inference processes.
models.py: Contains model architecture and utilities for loading and saving models.
data_handling.py: Utilities for data processing and loading.
evaluator.py: Functions for evaluating model performance.
configs.py: Configuration and argument parsing.
