# A Unified Approach to Interpreting Model Predictions

## Additive Feature Attribution Methods

- For *simple model*, the best explanation is the model itself.

- For *complex model* (such ensemble methods and deep neural networks), is better use a simpler **explanation model**: any interpretable approximation of the original model.

**Settings**:

- *local methods*

- $f$: original model
    * $x$: input
- $g$: explanation model
     * $x'$: simplified input s.t. $x = h_x(x')$
     * $g(x') \approx f(h_x(x'))$

**Additive feature attribution methods**: have an explanation model that is a linear function of binary variables
$$
g(z') = \phi_0 + \sum_{i=1}^{M} \phi_i z'_i,
$$
where:
- $z' \in {0,1}^M$
- $ \phi_i \in \mathbb{R}$
- $M$ is the number of simplified input features

The above equation is essentially a summing the effects

### LIME

This method implements the equation above exactly. The mapping function, that maps a binary vector of the interpretable inputs $x'$ into the original input space, could be of differente types (specific for the domain of interest, like words or images).

LIME, in order to find $\phi$, minimizes the following objective function
$$
\xi = \arg \min_{g \in \mathcal{G}} L(f, g, \pi_{x'}) + \Omega(g).
$$
where:
- $L$ is the squared loss (over a set of sample in the simplified input space), weighted by $\pi_{x'}$
- $\pi_{x'}$ local kernel
- $\Omega$ penalization factor applied to model $g$

### DeepLIFT

Here the mapping function converts 1 into original value and 0 into *reference value*: it represents a typical uninformative background value for the feature.

$$
\sum_{i=1}^{n} C_{\Delta x_i \Delta o} = \Delta o
$$
where:
- $o = f(x)$ is the model output
- $\Delta o = f(x) - f(r)$
- $\Delta x_i = x_i - r_i$
- $r$ is the reference input
To matching the additive feature attribution methods' equation, we need to set $\phi_i = C_{\Delta x_i \Delta o}$ and $\phi_o = f(r)$

### Layer-Wise Relevance Propagation

This method is like DeepLIFT but in binary vectors, the 0's value is mapped to 0 in original feature space and not to a reference value.

### Classic Sahpley Value Estimation

The three methods above use:
- **Sahpley regression values**, they are feature importances for linear models

Considering $F$ as set of all features for a model:
- retraining the model on $S \subseteq F$

<!-- DUBBIO -->
- It assigns an importance value to each feature that represents the effect on the model
prediction of including that feature. To compute this effect, a model $f_{S∪{i}}$ is trained with that feature
present, and another model $f_S$ is trained with the feature withheld. ????????????

- the prediction of two models are compared on the current input $f_{S\cup\{i\}}(x_{S\cup\{i\}}) - f_S(x_S)$ where $x_S$ represents the values of the input features in the set $S$.

- ??? (Legato al dubbio) the preceding differences are computed for all possible subset $S \subseteq F \setminus \{i\}$

(See the equation in the paper)

Here the mapping function converts, into the original space, 1 means the input is included in the model, 0 otherwise. Setting then $\phi_0 = f_{\emptyset}(\emptyset)$, the Shapley regression values match the additive feature attribution methods' equation.

- **Shapley sampling values**: it is like Sahpley regression values (SRP) but:
    * applying sampling approximations to the SRP's equation
    * approximating the effect of removing a variable from the model by integrating over samples from the training dataset
    * this avoid the retraining of the model and are computed less than $2^{|F|}$ differences

- **Quantitative input influence** is a broader framework that addresses more than feature attributions.
However, as part of its method it independently proposes a sampling approximation to Shapley values that is nearly identical to Shapley sampling values. It is thus another additive feature attribution method.

## Simple Properties Uniquely Determine Additive Feature Attributions

1. **Local accuracy**: approximating $f$, given $x$, it is required that
$$
f(x) = g(x') = \phi_0 + \sum_{i=1}^M \phi_i x_i'
$$
where $x = h_x(x')$

2. **Missingness**: the features in the original input space, that are not present in the feature space of the simplified input, should have no impact.
$$
x_i' = 0 \implies \phi_i = 0
$$

3. **Consistency** ???

- **Theorem 1**:  only one possible explanation model $g$ follows Definition 1 and satisfies Properties 1, 2 and 3

## SHAP values

### Model-Agnostic Approximations

Here are assumed the feature independence to approximating conditional expectations. The SHAP values can be estimated directly by using:
- Shapley sampling values
- Quantitative Input Influence 

**Kernel SHAP (Linear LIME + Shapley values)**

LIME uses a linear explanation model to locally approximate $f$.

It choices for the parameters (listed in the paper) are made heuristically $\implies$ local *accuracy* and/or *consistency* are violated $\implies$ **Sharpley Kernel theorem**...

### Model-Specific Approximation

More faster than model-agnostic approximation

**Linear SHAP**: given linear models, if is assumed input feature independence, SHAP values can be approximated directly from the model's weight coefficients.

**Low-Order SHAP**

**Max SHAP**

**Deep SHAP (DeepLIFT + Shapley values)**

Combining DeepLIFT (recursively) and Shapley values: SHAP values are computed for smaller components of the network into SHAP values for the whole network.


# ACME

Taken SHAP as inspiration, AcME is a model-agnostic approach that are faster than tha Kernel SHAP variant. 
AcME does not require any retraining.
It provides similar global explanation to those provided by SHAP, in a fraction of the computation time.

## Global interpretability for regression tasks

1. Compute the baseline vector: mean values of the features
2. Replace the $j$-th variable with the quantile $q$, creating a new vector from the baseline one
3. With these new vectors as rows, compose the relative matrix
4. Compute predictions associated to each row
5. Calculate the standardized effect
6. Compute the score for each feature by averaging the magnitude of standardized effect

## Local interpretability for regression tasks

Similar procedure but the baseline vector is equal to the specific data point to be explained