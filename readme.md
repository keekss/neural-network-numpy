# Neural network using `numpy`

### Overview

Neural network with a variable number of hidden layers, neurons, and various training hyperparameters.

Training (`fit` method) attribute tables and progress graphs during training depending on `verbosity`. After training, "summary" stats are written to `.csv` format by default.

Later hyperparameter tuning is done with the `hyperopt` library.

Attribute | Value | Notes
:-- | :-- | :--
Hidden layer activations | ReLU | i.e. Rectified Linear Unit. `ReLU(x) = max(0,x)`
Output activation | Softmax | Normalized to prevent overflow
Loss metric | Categorical cross-entropy
Gradient descent update rule | Adam | i.e. Adaptive Movement Estimation
Overfitting prevention | Dropout | Dropout rates can be manual or ranged and reshuffled (see `fit` function)

Weights are stored as member variables via matrices.

## Initialization
### General structure
Parameter | Meaning | Fashion MNIST Value
:-- | :-- | --:
`inputs` | Number of input features | `784`
`outputs` | Number of outputs (predictions) | `10`

### Hidden neurons: shaped or manual

Number of hidden layers & height (number of neurons) at each layer will be determined by the  parameters `manual_heights`, `h_layers`, `shape`, `max_height`, and `shrink_factor`.

The hidden layers will either be manually structured or "shaped" (discussed below).

Manual: If `manual_heights` is not `None`, the number of hidden neurons at each layer will be entries in the list `manual_heights`, and the other hidden-neuron-related parameters are _ignored_.

Shaped: Otherwise, the number of hidden neurons at each layer will be generated according to a "shape":

`shape` can take one of three possible values:

Shape | Meaning
:-- | :--
`'flat'` | All hidden layers have the same number of hidden neurons
`'contracting'` | Hidden layers will linearly _decrease_ in height from left to right, with the first hidden layer having `max_height` neurons and the last layer having `(max_height)(shrink_factor)` neurons
`'expanding'` | Same structure as `'contracting'`, except from right to left (i.e. the last hidden layer is the tallest)

The number of hidden layers will be `h_layers`, and exact heights will be generated by `np.linspace`.

For example, `h_layers=3`, `shape='contracting'`, `max_height=500`, `shrink_factor=0.5` will generate a heights list of `[500,375,250]`.



# Training (`fit` function)

## General hyperparameters
| Parameter(s) | Meaning | Type | Notes
| :-- | :-- | :-- | :--
| `X`, `y` | Training data | `np.array` | Examples stored as rows
| `x_val`, `y_val` | Validation data | `np.array` | Examples stored as rows
| `lr` | Learning rate | `float` | Fraction by which weight adjustments (i.e. negative loss gradients) are multiplied during fitting for each mini-batch to update weights
| `batch_size` | Mini-batch size | `int` | The number of examples trained simultaneously before weights are adjusted


## Dropout: type
The model will be trained according to one either type of dropout (not both):

| Dropout type | Associated param | Type | Notes
| :-- | :-- | :-- | :--
| Manual | `keep_rates_manual` | `list` | One `keep_rate` for each hidden layer
| Ranged | `keep_rates_range` | `list` | `min` and `max` `keep_rate`s to be randomly set for each layer per `np.random.uniform`

## Dropout: reshuffling
`keep_rates_reshuffle` determines reshuffling status for all of training.
| Reshuffling | Dropout type | Meaning
| :-- | :-- | :--
| Yes | Manual | `keep_rates` is reordered randomly for each mini-batch
| Yes | Ranged | `keep_rates` is re-rolled per `np.random.binomial` for each mini-batch
| No  | Manual | `keep_rates` remains as passed
| No  | Ranged | `keep_rates` is not modified after instantiation at the start of training (per `np.random.binomial`)

Note: to avoid ambiguity, values are stated as `keep_rate`, i.e. the odds of a given neuron being "kept" during training.  Neurons are dropped according to a binomial distribution during training.


## Early stopping tolerance

The model incurs a "strike" when validation loss increases between epochs.  The limits, `total_strikes` and `total_strikes_consecutive`, can be specified.  If either of these conditions limits are exceeded, training stops.
