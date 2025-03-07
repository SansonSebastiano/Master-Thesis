# Counterfactual Explanations (Chapter 9.3 - Interpretable ML - Molnar)

*Counterfactual*: is the ability of imagining a hypotetical reality that contradicts the observed facts. Hence in ML this means that the input feature values cause the prediction.

**A counterfactual explanation of a prediction describes the smallest change to the feature values that changes the prediction to a predefined output.**

This method are appliable both for *model-agnostic* and *model-specific* (this chapter is focused only on the first one)

Counterfactuals are **contrastive** and **selective**.

*Rashomon effect*: same results but contraditional explanations -> solution:
- reporting all conterfactual explanations
- adopting a criterion to evaluate for selecting the best one:
    1. *A counterfactual instance produces the predefined prediction as closely as possible.* Generally it is not possible: relaxing the above condition.
    2. *A counterfactual should be as similar as possible to the instance regarding feature values.* For measuring this quality it is sufficient to measure the distance between two instance $ L_0 $
    3. *Multiple diverse counterfactual explanations*: trying different ways to get the desired outcome
    4. *A counterfactual instance should have legal feature values*

## Generating counterfactual explanations

Defining a *loss function* and an *optimization algorithm*. We focus on two methods.

### Method by Wachter et al.

**Loss function**

$$L(x, x', y', \lambda) = \lambda \cdot (\hat{f}(x') - y')^2 + d(x,x')$$

where:
- $d(x,x') = \sum_{j=1}^{p} \frac{|x_j - x_j'|}{MAD_j}$ is the Manhattan distance weighted with $MAD$
    - $MAD_j = median_{i \in {1,\dots,n}}(|x_{i,j} - median_{l \in {1, ..., n}(x_{l,j})}|)$ 
- $\lambda$ parameter that balances the distance in prediction against the distance in features values.
    - higher $\lambda$ means that we want small $(\hat{f}(x') - y')^2$
    - lower $\lambda$ means that we want $x'$ similar to $x$

How to choose the right $\lambda$? By selecting a tollerance $\epsilon$ such that $|\hat{f}(x') - y'| \leq$ $\epsilon$

**Optimization algorithm**

- Any suitable optimization algorithm
- Gradient-based methods if the gradient is accessible 

$$ \arg \min_{x'} \max_{\lambda} L(x,x',y',\lambda)$$

Procedure:

1. Select:
    * an instance $x$ to be explained
    * the desired $y'$
    * a tollerance $\epsilon$
    * a low initial $\lambda$
2. Sample a random instance as initial counterfactual
3. Optimize the loss with the initial sample counterfactual
4. While $|\hat{f}(x') - y'| > \epsilon$
    * increase $\lambda$
    * optimize the loss with the current counterfactual 
    * return the counterfactual that minimizes the loss
5. Repeat steos 2-4 and return the list of counterfactuals that minimize the loss

*Disadvantage*: take into account only the first two requirements

### Method by Dandl et al.

## Advantages

## Disadvantages