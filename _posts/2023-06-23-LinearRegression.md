---
title: Linear Regression
date: 2023-06-23 10:00:00 +0800
categories: [Machine Learning, Regression]
---
## The Usual Introduction
Linear Regression is typically thought of in the following terms: We have a response variable $Y$ and a set of $n$ predictor variables represented as $X$ = $(x_1, x_2, \dots, x_n)$ where each $x_i$ represents a vector of observations for a particular predictor variable. 
When we stack these vectors, we get the matrix $X \in \mathbb{R}^{m \times n}$, where $m$ is the number of observations. 
Let 
$$
f(X) = \sum_{i=1}^n \beta_i x_i + \beta_0
$$
then linear regression assumes that there exists an approximate linear relationship between $Y$ and $X$ such that
$$
Y \approx f(X) + \epsilon
$$
where $\boldsymbol{\beta} = (\beta_0, \beta_1, \dots, \beta_n)^T \in \mathbb{R}^{n+1}$ is a vector of unknown coefficients and $\epsilon$ is the error. We also assume that $\epsilon \sim \mathcal{N}(0, \sigma^2)$ so that $\mathbf{P}(Y \mid X,\boldsymbol{\beta} ) \sim \mathcal{N}(X \boldsymbol{\beta}, \sigma^2)$.

This means that for infinitely many data points, the sample average of the errors $\epsilon \rightarrow 0$ by the Law of Large Number using the fact that the errors are assumed to be iid (independent and identically distributed) with mean 0.

The term "linear" in linear regression refers to the fact that the response variable $Y$ is a linear in the coefficients or parameters. This means that even if 
$$
X = (x_1, x_2, x_2^2, x_2^3)
$$ then the model is still linear. Note that it is not necessary that we assume the error $\epsilon$ to be normally distributed but by doing so, we can derive some nice properties on the solution of the linear regression problem which we will see later.

## Least Squares
One of the most (if not the most) popular method of parameters estimation for linear regression is the method of least squares. The idea is to find the parameters $\boldsymbol{\beta}$ that minimizes the residual sum of squares
$$
RSS(\boldsymbol{\beta}) = \sum_{i=1}^n (y_i - f(x_i))^2.
$$
The idea is in itself quite intuitive. It finds the best linear fit with respect to the squared errors. Of course, one may use other loss functions such as the absolute value loss function $L_1$ called Least Absolute Deviation (LAD) which is more robust against outliers but does not have a closed form solution. By using the squared loss function, we can easily derive the solution to the least squares problem as 
$$
\hat{\boldsymbol{\beta}} = (X^T X)^{-1} X^T Y.
$$
Note that this solution is only valid if $X$ has full column rank, i.e, its columns are linearly independent then $X^T X$ is positive definite and is invertible. This means that each feature of the model is assumed to be linearly independent of each other.    
For notation purposes, let $X = (x_0, x_1, \dots, x_n)$ where each $x_i$ are column vectors with $x_0 = 1$ (for the bias) then the estimated $\hat{y}$ is given by 
$$
\hat{y} = X \hat{\boldsymbol{\beta}} = X (X^T X)^{-1} X^T y
$$
where the matrix $ (X^T X)^{-1} X^T$ is called the orthogonal projection matrix. This matrix projects the vector $Y$ onto the column space of $X$ which is the space spanned by the columns of $X$. This means that the vector $\hat{y}$ is the projection of $y$ onto the column space of $X$. For more information on orthogonal projection, please visit [this](https://math.libretexts.org/Bookshelves/Linear_Algebra/Interactive_Linear_Algebra_(Margalit_and_Rabinoff)/06%3A_Orthogonality/6.03%3A_Orthogonal_Projection) link.

## Maximum Likelihood Estimation
### Likelihood Function
A more probabilistic approach to derive the parameters of the linear regression model is the method of maximum likelihood estimation. The idea is to find the parameters $\boldsymbol{\beta}$ that maximizes the likelihood function
$$
\mathcal{L}(\boldsymbol{\beta}) = \prod_{i=1}^n P(y_i \mid x_i, \boldsymbol{\beta}).
$$
In a more general setting, we want to find the parameters $\hat{\boldsymbol{\theta}}$ such that 
$$
\hat{\boldsymbol{\theta}} = \arg \max_{\boldsymbol{\theta}} P(\mathcal{D} \mid \boldsymbol{\theta})
$$
where $P(\mathcal{D} \mid \boldsymbol{\theta})$ is the likelihood, the probability of the data $\mathcal{D}$ given the parameters $\boldsymbol{\theta}$. **Generally, we wants the parameters that best predicts the observed data, hence the name maximum likelihood estimation**. 

As an example. Let's say you have a bag of 100 marbles, some of which are red and some are blue. You draw 10 marbles from the bag without looking, and 7 of them are red. If you have two theories: Theory A suggests the bag has 70 red marbles and 30 blue marbles, while Theory B suggests it has 50 of each color, the "likelihood" of each theory can be calculated based on the observed data (7 out of 10 marbles drawn were red). In this case, Theory A has a higher likelihood because it better predicts the observed data.

### Log Likelihood Function
Since the likelihood function is a product of probabilities which makes it difficult to operate on (taking derivatives, etc). We can instead work with the log likelihood function which transforms the product into a sum. This does not change the location of the maximum of the likelihood since log is a monotonic function. Hence, we can write the log likelihood function as
$$
\log \mathcal{L}(\boldsymbol{\beta}) = \sum_{i=1}^n \log P(y_i \mid x_i, \boldsymbol{\beta})
$$
where, by the Gaussian assumption, 
$$
P(y_i \mid x_i, \boldsymbol{\beta}) = \frac{1}{\sqrt{2 \pi \sigma^2}} \exp \left( - \frac{(y_i - f(x_i))^2}{2 \sigma^2} \right).
$$
Substituting this into the log likelihood function, we get
$$
\log \mathcal{L}(\boldsymbol{\beta}) = - \frac{n}{2} \log (2 \pi \sigma^2) - \frac{1}{2 \sigma^2} \sum_{i=1}^n (y_i - f(x_i))^2.
$$
Note that the residual sum of squares appears in the log likelihood so by maximizing the log likelihood by taking the derivative, you will obtain the same solution as the least squares method. **Thus the least squares method is equivalent to the maximum likelihood estimation under the assumption that the errors are normally distributed**.