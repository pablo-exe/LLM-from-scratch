# LLM from scratch [Work in progress]

Implementing an LLM from scratch from data preparation to SFT.

The idea is to keep the GPT-2 architecture: a transformer decoder with masked self-attention and classic learned positional embeddings.

So far, I’ve done some initial attention experiments in [Transformer.ipynb](https://Transformer.ipynb): embeddings, Q, K and V projections, scaled dot-product attention, and a residual connection. I still need to split the attention into multiple heads and add the causal mask. Everything is still at a very early stage, and I’ll keep building it up gradually.

I want to go through the entire process while keeping this architecture as the foundation. For now, it's a work in progress.

**## Structure**

- `notebooks/`: a step-by-step exploration of data, attention, architecture, pretraining, and fine-tuning.
- `src/llm_from_scratch/`: tokenizer and dataset, causal attention, GPT, training, generation, classification, and LoRA.
- `scripts/`: entry points for training, generating, and fine-tuning the model.
- `tests/`: shape and causality tests for attention and the model.
