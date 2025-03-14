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
However, as part of its method it independently proposes a sampling approximation to Shapley values
that is nearly identical to Shapley sampling values. It is thus another additive feature attribution
method

## Simple Properties Uniquely Determine Additive Feature Attributions