---
layout: post
title: "The Bias–Variance Tradeoff, Carefully"
date: 2026-05-20 12:00:00
description: A rigorous derivation of the bias-variance decomposition and what it actually tells us about modern overparameterized models.
tags: [math, tutorial, deep-learning]
categories: []
related_posts: false
toc:
  sidebar: left
---

The bias–variance tradeoff is one of the oldest ideas in statistics, but it is also one of the most frequently misunderstood — especially in light of modern deep learning, where highly overparameterized models generalize well despite having near-zero training loss.

## The Decomposition

Let $$y = f(x) + \epsilon$$ where $$\epsilon \sim \mathcal{N}(0, \sigma^2)$$ is irreducible noise. Given an estimator $$\hat{f}$$ learned from a dataset $$\mathcal{D}$$, the expected squared error at a point $$x$$ decomposes as:

$$\mathbb{E}_{\mathcal{D},\epsilon}\!\left[(y - \hat{f}(x))^2\right] = \underbrace{\left(\mathbb{E}_{\mathcal{D}}[\hat{f}(x)] - f(x)\right)^2}_{\text{Bias}^2} + \underbrace{\mathbb{E}_{\mathcal{D}}\!\left[\left(\hat{f}(x) - \mathbb{E}_{\mathcal{D}}[\hat{f}(x)]\right)^2\right]}_{\text{Variance}} + \underbrace{\sigma^2}_{\text{Noise}}$$

**Proof sketch.** Let $$\bar{f} = \mathbb{E}_\mathcal{D}[\hat{f}(x)]$$. Expand $$(y - \hat{f})^2 = (y - \bar{f} + \bar{f} - \hat{f})^2$$ and use the fact that the cross term vanishes in expectation because $$\epsilon$$ is independent of $$\mathcal{D}$$, and $$\hat{f} - \bar{f}$$ is mean-zero over $$\mathcal{D}$$.

## The Classical Picture

In the classical view, model complexity controls a tradeoff:

- **High bias, low variance**: simple model (e.g., linear regression on nonlinear data). Consistently wrong but in a predictable way.
- **Low bias, high variance**: complex model (e.g., degree-100 polynomial). Gets training data right but fluctuates wildly with different samples.

This produces the familiar U-shaped test error curve as a function of model complexity.

## The Double Descent Phenomenon

Modern neural networks break this picture. As model size grows past the interpolation threshold (where the model can fit the training data exactly), test error often *decreases again* — a phenomenon called **double descent** (Belkin et al., 2019).

The classical U-curve is just the left portion. After the interpolation peak, the model has so many parameters that among all solutions that fit the training data, implicit regularization (e.g., from SGD or the inductive bias of the architecture) selects a smooth, low-variance one.

This suggests the decomposition is still valid, but the mapping from "model size" to "bias" and "variance" is more nuanced than the classical picture suggests: larger models can simultaneously achieve low bias *and* lower variance than slightly smaller models.

## Practical takeaways

1. The decomposition is exact — it is not an approximation or a heuristic.
2. Variance is reducible by ensembling or averaging over random seeds.
3. Bias is reducible by using a richer model class.
4. Neither tells you about computational cost or the quality of your optimization.

## Further reading

- Belkin et al. (2019), *Reconciling modern machine learning and the bias–variance tradeoff*: [arxiv.org/abs/1812.11118](https://arxiv.org/abs/1812.11118)
- Hastie, Tibshirani & Friedman, *The Elements of Statistical Learning* (free PDF) — Chapter 7 for the classical treatment.
