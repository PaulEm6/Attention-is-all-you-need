# Attention is all you need
Creating a decoder only transformer with self attention blocks based on "Attention is all you need" paper and Andrej Kaparthy Youtube tutorial

# Decoder-Only Transformer

This project implements a decoder-only transformer model based on the "Attention is All You Need" architecture proposed in the seminal paper by Vaswani et al. The implementation is guided by the YouTube tutorial by Andrej Kaparthy, providing a clear understanding of the transformer architecture and its components.

## Dependencies

- PyTorch

## Overview

The main focus of this project is to implement and explore the decoder-only variant of the transformer model. Unlike the original transformer architecture, which consists of both encoder and decoder components, the decoder-only transformer is designed for tasks where only the decoder is needed, such as language generation.

## Features

- Implementation of the decoder-only transformer architecture
- Word-based tokenization for input data
- Experimentation with different datasets, including:
  - Lyrics of an artist
  - Papers authored by the user
- Evaluation and improvement of model performance on various datasets
- Fine-tuning the model for question/response alignment

