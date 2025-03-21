# Counterfactual Explanations (Chapter 9.3 - Interpretable ML - Molnar)

*Counterfactual*: is the ability of imagining a hypotetical reality that contradicts the observed facts. Hence in ML this means that the input feature values cause the prediction.

**A counterfactual explanation of a prediction describes the smallest change to the feature values that changes the prediction to a predefined output.**

This method are appliable both for *model-agnostic* and *model-specific* (this chapter is focused only on the first one)

**Formal Definition**: given a classifier $b$ and its outcome $y = b(x)$, a counterfactual explanation consists of an instance $x'$ such that $b(x') \neq y$ and the difference between $x$ and $x'$ is *minimal*

Counterfactuals are **contrastive** and **selective**.

*Rashomon effect*: same results but contraditional explanations -> solution:
- reporting all conterfactual explanations
- adopting a criterion to evaluate for selecting the best one (**requirements**):
    1. *A counterfactual instance produces the predefined prediction as closely as possible.* Generally it is not possible: relaxing the above condition.
    2. *A counterfactual should be as similar as possible to the instance regarding feature values.* For measuring this quality it is sufficient to measure the distance between two instances $L_0$ (*minimality*)
    3. *Multiple diverse counterfactual explanations*: trying different ways to get the desired outcome (*diversity*)
    4. *A counterfactual instance should have legal feature values* (*plausibility*)

## Properties of counterfactual explanations

- **Validity**: $b(x) \neq b(x')$
- **Minimality**: $x'$ is *mimimal* iff $\nexists x''$ s.t. $|\delta_{x,x''}| < |\delta_{x,x'}|$ where $\delta_{.,.}$ is the distance
- **Similarity**: a counterfactual $x'$ should be similar to $x$, i.e. $d(x, x') < $ $\epsilon$ 
- **Plausibility**: Given a reference population $X$, a counterfactual $x'$ is plausible if the feature values in $x'$ are coherent with those in $X$. This practically means that the feature values of $x'$ should not be higher/smaller than those observable in $X$, and that $x'$ should not be labeled as an outlier with respect to the instances in $X$. Plausibility helps in increasing trust towards the explanation: it would be hard to trust a counterfactual if it is a combination of features that are unrealistic with respect to existing examples
- **Discriminative**: A counterfactual $x'$ should show a high discriminative power for recognizing the reasons for the decision outcome
- **Actionability**
- **Causality**
- **Diversity**

## Properties of counterfactual explainers

- **Efficiency**
- **Stability**
- **Fairness**

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
5. Repeat steps 2-4 and return the list of counterfactuals that minimize the loss

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

This loss want to optimize all four terms simultaneously by using **Nondominated Sorting Genetic Algorithm**: denotes the fitness of a counterfactual by its vector of objective values $(o_1, o_2, o_3, o_4)$, the lower the values, the "fitter" it is.

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

# Isolation Forest

[Paper](https://www.lamda.nju.edu.cn/publication/icdm08b.pdf)

## Isolation and Isolation Trees

It is a model-based methods that explicitly isolates anomalies instead of profiles the normal points.

It is an opposite approach to the classical ones like:

- *classication-based model*
- *clustering-based methods*

Isolating anomalies, we can exploit some quantitative properties:

- anomalies << normal instances
- have `(attribute-values)` very different from those of normal instances

This method construct a tree, **Isolation Tree** (the opposite of the classical (Profiling) Tree),in which the anomalies are isolated closer to the root, while the normal instance are isolated at the deeper end of the tree.

So can scale up to handle more larger data size and high-dimensional problem.

**Isolation Forest** = esembles of Isolation Trees; the variables are:

- `number_of_trees`
- `sub_sampling_size`

In general a *normal point* requires more (random) partitions to be isolated than *anomaly point* and the latter have path lengths shorter than normal instances.

**Isolation Tree**: let $T$ be a node of an isolation tree. $T$ is either an *external-node* with no child, or an *internal-node* with one test (i.e. a threshold) and exactly two daughter nodes
$(Tl,Tr)$. A *test* consists of an attribute $q$ and a split value $p$ such that the test $q<p$ divides data points into $Tl$ and $Tr$.

- It is a *proper binary tree* (each node have zero or two daughter)
- $n$ external nodes
- $n-1$ internal nodes 
- total number of nodes = $2n-1$

Stop conditions while building an iTree:

1. height limit reached
2. $|X|=1$
3. all data in $X$ have the same values

**Path Length** $h(x)$ is measured by the number of edges traversed to reach $x$ from the root.

<!-- DUBBIO 1-->
Deriving **anomaly score** (see DIFFI paper) from $h(x)$ is difficult, based on *maximum height* or *average height*

$c(x)$ needed to normalize $h(x)$, $c(x)$ is the average of $h(x)$ ????

## Characteristic of Isolation Trees

- Anomalies are identified by having shorter path lengts
- Each iTree in the ensemble acts as "expert" to target different anomalies.
- iForest works well with a partial model, i.e. without using the majority of dataset (which contains the normal instances), so with a small sample size.
- *Swamping* problem: more partitions required when a normal instance is too close to an anomaly.
- *Masking* is the existence of too many anomalies concealing their own existence. When an anomaly cluster is too large and dense, it causes:

    * increasing the number of partitions
    * longer paths for anomalies, so more difficult to be detected

Hence a sub-sampling could help to solve these problems, another demonstration that iForest works better with small sample size.

## Anomaly Detection using iForest

Two-stage process:

1. *Training* - building iTrees using sub-samples of the training set
2. *Testing* - test instances through iTrees to obtain an anomaly score for each instance

### Training stage

iTrees are constructed by recursively partitioning the given training set until:

- instances are isolated
- a specific tree height is reached

The data points which have shorter-than-average path lengths, those points are more likely to be anomalies.

*See the algorithms in the paper*

**Complexity**: $O(t\psi log\psi)$

### Test stage

An anomaly score $s$ is derived from the *expected path length* $E(h(x))$ for each test instance.

<!-- DUBBIO 2 -->

**Complexity**: $O(ntlgo\psi)$

### Evaluation

iForest outperforms ORCA, LOF and Random Forest, in term of AUC and in processing time.

Linear time complexity, with a low memory requirement.

With small $\psi$ the AUC converges very quickly. Then the processing time increases very modestly when $\psi$ increases enormously.

It has also a low processing time even in high dimensional data (with irrelevant attributes).

It works well also without outliers.

# Interpretable Anomaly Detection with DIFFI: Depth-base Isolation Forest Feature Importance

This paper proposes a solution to solving the main problem of iForest: *the lack of interpretability*

Remember that AD is unsupervised task.

DIFFI is: 
- model-specific w.r.t. iForest
- a global method, but there is also a local version.
- post-hoc method

For each iTree, will be considered only the boostrap sample, rather than the entire training set.

<!-- DUBBIO 3 -->
Intuition 2???

## ICC (Induced Imbalance Coefficient)

$$
\lambda(v) = \begin{cases}
0, & \text{if } n_l(v) = 0 \text{ or } n_r(v) = 0 \\
\tilde{\lambda}(v), & \text{otherwise}
\end{cases}
$$
where
$$
\tilde{\lambda}(v) = g \left( \frac{\max(n_l(v), n_r(v))}{n(v)} \right)
$$
and $g(\cdot)$ is a scaling function mapping its input into the interval $[0.5, 1]$.
In the first case, the split results useless, while in the second one there is an *isolating splitting*: this happens when either the left or the the right child receives exactly one data point and is assigned the highest possible IIC, i.e. 1.

## CFIs (Cumulative Features Importance) update

- $I_I$: CFI for inliers
- $I_O$: CFI for outliers

**Both are $p$-dimensional vector, where the $j$-th component represent the CFI for $j$-th feature.**
The update of the CFI is done in additive fashion.

The procedure (for $I_I$, but is the analogous for $I_O$) is iterating over all iTree $t$ in the iForest, over the $\mathcal{P}_{I,t}$ (the subset of predicted inliers).

Then for generic $\boldsymbol{x}_I\in\mathcal{P}_{I,t}$, we iterate over the internal nodes in its path $Path(\boldsymbol{x}_I, t)$, then update the $j$-th component of $I_I$, that it the splitting feature associate to a generic internal node.

**(See the update rule in the paper)**

## GFIs computation

Problem: selecting a splitting feature randomly cause an unfair CFI high value, because it is selected more frequently than others.

Define so a *features counter*, $C_I$ for inliers and $C_O$ for outliers: a $p$-dimensional vector in which the $j$-th component represents how many times such feature is appeared while updating CFIs.

$$
GFI = \frac{I_O/C_O}{I_I/C_I}
$$

**(See the algorithm in the paper)**

## Local-DIFFI

For the intepretation of individual predictions, using the similar procedure above is quite impossible (the reasons are written in the paper).

So given a predicted outliers $\boldsymbol{x_O}$:
$$
LFI(\boldsymbol{x_O}) = \frac{I_O^{loc}}{C_O^{loc}}
$$
where:
- $C_O^{loc}$ is the features counter
- $I_O^{loc}$ is the CFI associated to the predicted outliers. 

**(See  the update rule in the paper)**

## Unsupervised feature selection with global DIFFI

DIFFI can be exploited to perform feature selection in the context of AD problems.

The procedure consits in training $N_{fs}$ different instances of iForest, obtained with the same training set. The global DIFFI scores associated with each instance of iForest are then aggregated to define a ranking on the features:

1. Define $S_{agg} \in \mathbb{R}^p$, with $p$ as the number of features
2. For each iForest:
    * sorting DIFFI scores in decreasing order
   4 * update $S_{agg}$
3. $S_{agg}$ is then used to define a ranking over the features.

**(See more details about the above procedure in the paper)**

The procedure based on DIFFI requires
minimal – if any – hyperparameters tuning: the only hyperparameters
are inherited from the underlying proxy model.

# Optimal Counterfactual Explanations in Tree Ensembles

This paper proposes:

- Efficient mathematical models to search for counterfactual explanations in tree ensembles. Applicable to heterogeneus datasets.

- Integration with iForest for conterfactual explanations

- Extensions of the (flexible) model that can be adapted to each specific situation + *plausibility and actionability* (i.e. action that is possible to do to change the outcome)

## Background

### Mixed Integer Programming

### Counterfactual Explanations in Tree Ensembles

Given a training set $\{{\mathbf{x}_k, c_k}\}_{k=1}^n$ in which $\mathbf{x}_k \in \mathbb{R}^p$ is a sample with $p$-dimensional feature vector and a class $c_k \in C$.

A tree ensemble $\mathcal{T}$ learns a set of tree $t \in \mathcal{T}$ returning then a class probability, in which the corresponding class should be the one that maximizes the weighted sum of probabilities: $F_\mathcal{T}(\boldsymbol{x}) = \arg \max_c \sum w_t F_{tc}(\boldsymbol{x})$, where $F_{tc} : \mathcal{X} \rightarrow [0, 1]$. 

Considering than the following optimization problem:
$$
\min f_{\hat{x}}(\mathbf{x}) \\
\text{s.t.} \quad F_{\mathcal{T}}(\mathbf{x}) = c^* \\
\mathbf{x} \in X^{\text{P}} \cap X^{\text{A}}
$$
in which $\hat{\boldsymbol{x}}$ is the origin data point and $c^*$ is the desired prediction and $f_{\hat{x}}$ is a separable convex cost that represents
how difficult it is to move from $\hat{x}$ to $x$.
While polytopes $X^P$ and $X^A$ represent the space of *plausible* and *actionable* counterfactual explanations (CE) respectively.

- **Plausibility**: this coinstraint should ensure that a CE respects the structure of the data and that it is located in a region that has a sufficiently large density examples (exploiting the iForest)

- **Actionability**: concern the trajectory between $\hat{\boldsymbol{x}}$ and $\boldsymbol{x}$

## Methodology

1. Describing the variables and constraints that characterize the branches (of the tree) taken by the counterfactual example.

2. Including additional variables and constraints, modelling the counterfactual example's feature values and ensuring compatibility with all the branches choices.

### Branch Choices

**(See all the sets/variables definition in the paper)**

$$
\lambda_{td} = \begin{cases}
1 & \text{if the counterfactual descends towards the left branch} \\
0 & \text{otherwise}
\end{cases}
$$

### Feature consistency with the splits

The proposed method could be applied to heterogeneous dataset

- Numerical features

- Binary features

- Categorical features

### Objective function

$$
f(\mathbf{x}, \boldsymbol{\mu}, \boldsymbol{\nu}) = f^{\text{N}}(\boldsymbol{\mu}) + f^{\text{B}}(\mathbf{x}) + f^{\text{C}}(\boldsymbol{\nu})
$$

For *binary* and *categorical* features, since the corresponding model give direct access to the values:

- $f^{\text{B}}(\mathbf{x}) = \sum_{i \in I_{\text{B}}} (c_i^{\text{TRUE}} x_i + c_i^{\text{FALSE}} (1 - x_i))$

- $f^{\text{C}}(\boldsymbol{\nu}) = \sum_{i \in I_{\text{C}}} \sum_{j \in C_i} c_i^j \nu_i^j.$

While for *numerical* features can be accessed directly/indirectly.

This model gives an extensible framework for efficiently modeling most existing data types, decision-tree structures, and objectives

### Domain knowledge and actionability constraints

### iForest for Plausability

Isolation forests are trained to return an outlier score for any sample, inversely proportional to its average path depth within a set of randomized trees grown to full extent on random sample subsets. Therefore, constraining this average depth controls the outlier score (and consequently the plausibility) of the counterfactual explanation.


# FOCUS: Flexible Optimizable Counterfactual Explanations for Tree Ensembles

Previously the problem of counterfactual explanations was seen as a gradient-based optimization task, therefore only for differentiable models. This paper proposes to extend this to non-differentiable models, such as ensemble trees, exploiting probabilistic model.

**Optimal counterfactual explanations**: the *minimal* changes to the input required to change the outcome.

*Counterfactual explanations are based on counterfactual examples*: generated instances that are close to an existing instance but have an alternative prediction.

The difference between the original instance and the counterfactual example is the counterfactual explanation.

The method proposed is:
- effective
- efficient
- reliable
- model-specific

*Adversarial examples* (*fool* the model) are a type of counterfactual example (*explain* the model), but with incorrect alternative prediction.

In the literature were been proposed differentiable versions of the ensemble tree, however they are not optimal for counterfactual explanations.

## Problem definition

*(See in the paper)*

## Method

### Loss function definition

$$
\tilde{\mathcal{L}}(x, \bar{x} | f, \tilde{f}, d) = \mathcal{L}_{pred}(x, \bar{x} | f, \tilde{f}) + \beta \cdot \mathcal{L}_{dist}(x, \bar{x} | d)
$$
approximates the complete loss $\mathcal{L}_{pred}(x, \bar{x} | f, d)$, where:
- $ \mathcal{L}_{pred}(x, \bar{x} | f, \tilde{f}) = \mathbb{1}\left[\arg\max_{y} f(y | x) = \arg\max_{y'} f(y' | \bar{x})\right] \cdot \tilde{f}(y' | \bar{x}) $ is the approximate prediction loss
- $\tilde{f}$ is the approximation, derived from the original model.
- $\beta \in $ a weight

Since we have the approximation, it can be used, by minimizing it, to find the optimal counterfactual example.

### Tree-based models and their approximation

To obtain the differentiable approximation $\tilde{f}$ of $f$, a probabilistic approximation of the original tree ensemble should be constructed.

1. Define the tree-based models (*see it in the paper*)

2. Then approximate them, exploiting *sigmoid* and *softmax* functions.
    
    * Hyperparameters and their effects:
        - $\sigma$ for sigmoid
        - temperature $\tau$  for softmax

### FOCUS

It can be applied to iForest?

**Flexible Optimizable CounterfactUal Explanations for Tree EnsembleS**

It takes as input:
- an instance $x$
- a tree-based classifier $f$
- two hyperparameters $\sigma$ and $\tau$ to create the approximation $\tilde{f}$

## Experiments

# Counterfactual explanations and how to find them: literature review and benchmarking

It is a survey 
