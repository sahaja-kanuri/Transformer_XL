# Transformer_XL

This repository contains a Python script for implementing the Transformer-XL model using Google Colab. 

The Transformer-XL model was introduced in the 2019 paper, ‘Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context’ by Dai et al. (https://arxiv.org/abs/1901.02860). 

What is it?
Transformer-XL (standing for extra long) is a state-of-the-art neural network architecture designed for sequence modeling tasks, such as language modeling and text generation. 

How is it different from the standard Transformer architecture?
In a standard Transformer architecture, the model is trained on text segments of fixed lengths, without any information flow across sequences. This limits the possibility of long-term dependency on prior text beyond the fixed context length. The model additionally suffers from the problem of ‘context fragmentation’, which refers to a model lacking necessary contextual information required to well-predict the first few tokens. This arises because of how the context was selected, usually without respecting sentence or semantic boundary.

Transformer-XL overview:
The Transformer XL introduces the following 2 modifications:
1. It introduces the notion of recurrence to the original Transformer architecture. Here, we reuse the hidden states from previous segments which is known as the extended context. This serves as memory for the current segment, thereby building the recurrent connection between the sequences. This results in modelling long-term dependency and also, resolves the problem of context fragmentation.
2. It introduces a novel positional encoding scheme known as the relative positional embeddings, instead of using absolute positional encodings. The necessity for this arises in order to enable state reuse and to avoid temporal ambiguity. 

Model specifications: 
Increase ‘max_iters’ from 50 to perhaps 5000 if you have access to a better GPU to train the model for more iterations and to thereby obtain a smaller cross-entropy loss, to see better results with the language model for text generation.

The implementation is done in Tensorflow using Keras. It uses the Wikitext103 dataset, and is a word-level language model. We use a pretrained Tokenizer from HuggingFace for the Transformer-XL model. The specifications of the model are initialized in 1st Chapter: Initialising. The model was trained using a L4 GPU in Google Colab. 
