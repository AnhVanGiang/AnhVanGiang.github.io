---
title: Linear Regression
date: 2023-06-23 10:00:00 +0800
categories: [Machine Learning, Regression]
---
## Assumptions
Given a response variable $Y$ and a set of $n$ predictors $X$ = $(x_1, x_2, \dots, x_n) \in \mathbb{R}^{n}$, linear regression assumes that there exists a linear relationship between $Y$ and $X$:
$$
Y = X^T \boldsymbol{\beta}+ \epsilon
$$
where $\boldsymbol{\beta} = (\beta_0, \beta_1, \dots, \beta_n)^T \in \mathbb{R}^n$ is a vector of unknown coefficients and $\epsilon$ is the error. 
We also assume that $\epsilon \sim \mathcal{N}(0, \sigma^2)$ so that $\mathbf{P}(Y \mid X,\boldsymbol{\beta} ) \sim \mathcal{N}(X^T \boldsymbol{\beta}, \sigma^2)$. 