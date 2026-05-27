---
layout: post
title: "Attention Is All You Need — a close reading"
date: 2026-05-01 12:00:00
description: A careful walk-through of the original Transformer paper, with derivations and intuitions for the attention mechanism.
tags: [deep-learning, nlp, tutorial]
categories: []
related_posts: false
toc:
  sidebar: left
---

The 2017 paper *Attention Is All You Need* (Vaswani et al.) replaced recurrence with self-attention and became the backbone of nearly every modern NLP system. This post walks through the key ideas carefully, with derivations.

## Scaled Dot-Product Attention

Given queries $$Q \in \mathbb{R}^{n \times d_k}$$, keys $$K \in \mathbb{R}^{m \times d_k}$$, and values $$V \in \mathbb{R}^{m \times d_v}$$, the attention function is:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

The $$\sqrt{d_k}$$ scaling factor is often glossed over, but it matters: without it, dot products grow large in magnitude as $$d_k$$ grows, pushing the softmax into regions with very small gradients.

### Why dot products grow

If $$q$$ and $$k$$ are independent random vectors with zero mean and unit variance components, then:

$$q \cdot k = \sum_{i=1}^{d_k} q_i k_i \implies \mathbb{E}[q \cdot k] = 0, \quad \text{Var}[q \cdot k] = d_k$$

So the standard deviation scales as $$\sqrt{d_k}$$. Dividing by $$\sqrt{d_k}$$ restores unit variance, keeping softmax in a well-behaved regime.

## Multi-Head Attention

Instead of computing a single attention function, the paper projects $$Q, K, V$$ into $$h$$ lower-dimensional subspaces, computes attention in each, and concatenates:

$$\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)\, W^O$$

where $$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$.

This lets the model attend to information from different representation subspaces simultaneously — something a single wide attention head cannot do as effectively.

## Positional Encoding

Self-attention is permutation-equivariant: if you shuffle the tokens, the outputs shuffle identically. To inject sequence order, the paper adds sinusoidal positional encodings to the token embeddings:

$$PE_{(pos, 2i)} = \sin\!\left(\frac{pos}{10000^{2i/d}}\right), \quad PE_{(pos, 2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d}}\right)$$

The key property is that for any fixed offset $$k$$, $$PE_{pos+k}$$ is a linear function of $$PE_{pos}$$, making it easy for the model to reason about relative positions.

## What to read next

- The original paper: [arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
- Jay Alammar's illustrated transformer for visual intuition
- *Flash Attention* (Dao et al., 2022) for the IO-efficient implementation that makes modern LLM training practical
