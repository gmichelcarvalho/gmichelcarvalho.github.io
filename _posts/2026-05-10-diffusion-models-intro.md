---
layout: post
title: "Diffusion Models from First Principles"
date: 2026-05-10 12:00:00
description: Understanding score-based generative models and DDPMs through the math, not just the intuition.
tags: [deep-learning, tutorial, math]
categories: []
related_posts: false
toc:
  sidebar: left
---

Diffusion models have become the dominant paradigm for image (and now audio, video, and 3D) generation. This post builds up the theory from scratch.

## The Forward Process

The forward process gradually corrupts data $$x_0 \sim q(x_0)$$ by adding Gaussian noise across $$T$$ steps:

$$q(x_t \mid x_{t-1}) = \mathcal{N}\!\left(x_t;\; \sqrt{1-\beta_t}\, x_{t-1},\; \beta_t I\right)$$

where $$\{\beta_t\}_{t=1}^T$$ is a fixed variance schedule. A crucial property is that we can sample $$x_t$$ at any step directly from $$x_0$$ in closed form. Define $$\alpha_t = 1 - \beta_t$$ and $$\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$$. Then:

$$q(x_t \mid x_0) = \mathcal{N}\!\left(x_t;\; \sqrt{\bar{\alpha}_t}\, x_0,\; (1 - \bar{\alpha}_t) I\right)$$

Or equivalently: $$x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \epsilon$$, where $$\epsilon \sim \mathcal{N}(0, I)$$.

As $$t \to T$$, $$\bar{\alpha}_t \to 0$$, so $$x_T \approx \mathcal{N}(0, I)$$ — pure noise.

## The Reverse Process

The reverse process $$p_\theta(x_{t-1} \mid x_t)$$ is what we learn. The posterior $$q(x_{t-1} \mid x_t, x_0)$$ is tractable (since conditioning on $$x_0$$ makes everything Gaussian):

$$q(x_{t-1} \mid x_t, x_0) = \mathcal{N}(x_{t-1};\; \tilde{\mu}_t(x_t, x_0),\; \tilde{\beta}_t I)$$

where the mean $$\tilde{\mu}_t$$ and variance $$\tilde{\beta}_t$$ have closed-form expressions in terms of $$\alpha_t$$ and $$\bar{\alpha}_t$$.

The idea of DDPM (Ho et al., 2020): train a neural network $$\epsilon_\theta(x_t, t)$$ to predict the noise $$\epsilon$$ added to $$x_0$$. The simplified training objective is:

$$\mathcal{L}_{\text{simple}} = \mathbb{E}_{t, x_0, \epsilon}\!\left[\|\epsilon - \epsilon_\theta(x_t, t)\|^2\right]$$

## Score Matching Connection

The noise-prediction network is closely related to the **score function** $$\nabla_{x_t} \log q(x_t)$$. Since:

$$\nabla_{x_t} \log q(x_t \mid x_0) = -\frac{\epsilon}{\sqrt{1 - \bar{\alpha}_t}}$$

learning $$\epsilon_\theta$$ is equivalent to learning a scaled score. This connection to score-based generative models (Song & Ermon, 2019) unifies several lines of work under a common probabilistic framework.

## Sampling

Generation proceeds by iterating the learned reverse step from $$x_T \sim \mathcal{N}(0, I)$$ down to $$x_0$$. DDPM uses $$T = 1000$$ steps, making generation slow. DDIM (Song et al., 2020) introduces a deterministic sampler that works with far fewer steps (e.g., 50) by reinterpreting the forward process as a non-Markovian one.

## Further reading

- DDPM: [arxiv.org/abs/2006.11239](https://arxiv.org/abs/2006.11239)
- Score-based models: [arxiv.org/abs/1907.05600](https://arxiv.org/abs/1907.05600)
- Lilian Weng's excellent overview: [lilianweng.github.io/posts/2021-07-11-diffusion-models](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)
