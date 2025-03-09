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

It satisfies every four requirements by using the following loss: $$L(x,x',y',X^{obs}) = (o_1(\hat{f}(x'),y'), o_2(x,x'), o_3(x,x'), o_4(x',X^{obs}))$$
where each requirements is satisfied by following:

1. $ 
o_1(\hat{f}(x'), y') = 
\begin{cases}
0 & \text{if } \hat{f}(x') \in y' \\
\inf_{y' \in y'} |\hat{f}(x') - y'| & \text{else}
\end{cases} $ 
is the Manhattan metric ($L_1$ norm)

2. $ o_2 = \frac{1}{p} \sum_{j=1}^{p} \delta_G (x_j, x'_j) $ is the *Gower* distance, with $p$ being the number of features and $\delta_G(x_j, x'_j) = \begin{cases}
\frac{1}{\hat{R}_j}|x_j - x'_j| & \text{if } x_j \text{ numerical} \\
\mathbb{I}_{x_j \neq x'_j} & \text{if } x_j \text{ categorical}
\end{cases}
$ with $\hat{R}_j$ the observed value range, it helps to scale $\delta_G$ for all features between  0 and 1

3. $o_3(x, x') = ||x - x'||_0 = \sum_{j=1}^{p} \mathbb{I}_{x'_j \neq x_j}$

4. $o_4(x', \mathbf{X}^{obs}) = \frac{1}{p} \sum_{j=1}^{p} \delta_G(x'_j, x_j^{[1]})$, it infers how "likely" a data point is using $X^{obs}$, i.e. $x_j^{[1]} \in X^{obs}$

This loss want to optimize all four terms simultaneously by using **Nondominated Sorting Genetic ALgorithm**: denotes the fitness of a counterfactual by its vector of objective values $(o_1, o_2, o_3, o_4)$, the lower the values, the "fitter" it is.

Steps:

1.  Generate a group of counterfactual candidates.

    * Initialize each of them by changing randomly some features compared to our instance $x$ to be explained.

2. Evaluate each candidate using the four objective functions.

3. **Select** the fitter candidates (i.e. they are more likely to be selected)

4. The candidates selected are then pairwise **recombined** to produce children that are similar to them by:

    * averaging their numerical values
    * crossing over their categorical values

5. Perform a slightly mutation of the feature values of the children to explore the whole feature space.

6. Now we have two resulting groups: 

    * the parents
    * the children

   From each of them, we want only the best half, thus we perform:
    
    * *Nondominated Sorting Algorithm* to sorting the candidates according to their objective values

    * *Crowding Distance Sorting*, it are applied only when the first algorithm evaluate equally good the candidates, to sorting them according to their diversity

7. Given the ranking, we select the most promising and/or the most diverse half of the candidates. Repeating from this resulting set to achieve low objective values

## Advantages

- **The interpretation of counterfactual explanations is very clear**: If the feature values of an instance are changed according to the counterfactual, the prediction changes to the predefined prediction.

- The **counterfactual method does not require access to the data or the model**.

- **The counterfactual explanation method is relatively easy to implement**

## Disadvantages

- **Rashomon effect**