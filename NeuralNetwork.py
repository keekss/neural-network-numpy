import numpy as np
import pandas as pd
from   tqdm import tqdm
import time, os

from typing import List, Tuple, Optional, Union, Dict

class NeuralNetwork:
    def __init__(self, inputs: int, outputs: int):
        self.inputs  = inputs
        self.outputs = outputs

    def train(
            self, 
            # General
            X: np.ndarray, 
            y: np.ndarray, 
            x_val: np.ndarray, 
            y_val: np.ndarray, 
            lr: float = 3e-4, 
            batch_size: int = 2**9,
            # Dropout
            keep_rates_manual: Optional[List[float]] = None, 
            keep_rates_range: Optional[Tuple[float, float]] = None, 
            keep_rates_reshuffle: bool = False,
            # Early stopping tolerance
            max_consec_strikes: int = 2, 
            max_strikes: int = 2, 
            max_epochs: int = 100,
            # Progress printing
            verbose: int = 3, 
            save_summary: bool = True
    ) -> None:
        '''See markdown above for hyperparameter and function details'''

        batch_size = int(batch_size)

        # Generate or store keep_rate list
        assert not(keep_rates_manual and keep_rates_range), \
            'Cannot have manual & range-based dropout'
        keep_rates_descrip = 'None' # For summary dataframe
        if keep_rates_range:
            min, max = keep_rates_range
            self.keep_rates = [np.random.uniform(min, max) for _ in range(self.w_layers)]
            keep_rates_descrip = f'range: ({min}, {max})'
        elif keep_rates_manual:
            assert len(keep_rates_manual)==self.h_layers, \
                'Uneven dropout / hidden layers'
            self.keep_rates = keep_rates_manual
            keep_rates_descrip = f'manual: {keep_rates_manual}'
        else:
            self.keep_rates  = None
        # Accessed in prediction and gradients methods
        self.keep_rates_manual    = keep_rates_manual
        self.keep_rates_range     = keep_rates_range
        self.keep_rates_reshuffle = keep_rates_reshuffle

        # Store relevant hyperparameters for summary dataframe
        self.hyperparams = dict(
            h_layers    = self.h_layers,
            heights     = self.heights_descrip,
            lr          = lr,
            batch_size  = batch_size,
            keep_rates  = keep_rates_descrip,
            reshuff     = keep_rates_reshuffle,
            max_strikes = max_strikes,
            max_consec  = max_consec_strikes,
        )
        # Pin the current hyperparams to the top of printing,
        # which is otherwise cleared after each epoch
        self.print_cache = print_cache
        self.print_cache.extend([
            'Training w/ hyperparams:',
            pd.DataFrame(self.hyperparams, index=['']),
        ])
        self.adam_params['lr']  = lr
        self.verbose            = verbose
        start_time = time.time()

        # To return for hyperparameter tuning and summary dataframe
        self.min_loss         = np.inf
        self.min_loss_weights = None
        self.max_acc          = 0
        self.max_acc_weights  = None

        # Start strikes at zero
        total_strikes = 0
        consec_strikes = 0
        self.print_cache.extend([
            f"Total strikes:    0/{max_strikes}",
            f"Consec strikes:   0/{max_consec_strikes}",
        ])

        self.epoch_index = 0
        print('Pre-training evaluation:')
        self.eval_epoch(X, y, x_val, y_val, start_time)
        if self.verbose:
            self.print_progress()

        # Train until either `max_epochs` is reached or either
        # of the early stopping limits are reached
        while self.epoch_index <= max_epochs:
            self.epoch_index += 1

            self.train_epoch(X, y, batch_size)
            self.eval_epoch(X, y, x_val, y_val, start_time)

            # Access previous 2 losses and accuracies
            # for validation data
            prev_loss_val, curr_loss_val = \
                self.epoch_performance.loss_val[-2:]
            prev_acc_val, curr_acc_val = \
                self.epoch_performance.acc_val[-2:]

            # Update best stats if applicable
            if curr_loss_val < self.min_loss:
                self.min_loss = curr_loss_val
                self.min_loss_weights = np.copy(self.weights)
            if curr_acc_val > self.max_acc:
                self.max_acc = curr_acc_val
                self.max_acc_weights = np.copy(self.weights)

            # If starting to overfit
            if curr_loss_val > prev_loss_val:
                total_strikes   += 1
                consec_strikes  += 1
            else:
                consec_strikes = 0

            # Replace the last 2 lines of `print_cache` with the current strikes,
            self.print_cache[-2:] = [
                    f"Total strikes:    {total_strikes}/{max_strikes}",
                    f"Consec strikes:   {consec_strikes}/{max_consec_strikes}",
                ]

            if self.verbose:
                self.print_progress()

            # Early stopping: if total or consecutive allowed strikes
            # have been exceeded, stop training
            if (consec_strikes >= max_consec_strikes) \
                    or (total_strikes >= max_strikes):
                print(f'Stopped training after epoch {self.epoch_index}')
                if save_summary:
                    self.save_summary()
                return self.min_loss

        # By now, `max_epochs` has been reached
        print(f'Reached max epoch of {self.epoch_index}')

        # Write summary stats to file by default
        if save_summary:
            self.save_summary()

        # Return best loss value for hyperparameter tuning
        return self.min_loss

    def train_epoch(self, X, y, batch_size):
        '''Train all mini batches, until the model has been trained
        on all traning examples.  Forward- and backpropogate,
        updating weights after each mini batch.'''
        # Reshuffle training data
        X, y = self.shuffle_Xy(X, y)
        # Train mini batches until the end of the dataset is reached
        num_batches = np.ceil(X.shape[0] / batch_size) # ceil in case batch_size > num_rows
        batches_X   = np.array_split(X, num_batches)
        batches_y   = np.array_split(y, num_batches)

        print(f'Training epoch {self.epoch_index}:')
        iter = tqdm(zip(batches_X, batches_y), total = len(batches_X)) \
            if self.verbose else zip(batches_X, batches_y)
        for b_X, b_y in iter:
            self.train_mini_batch(b_X, b_y)

    def train_mini_batch(self, X, y):
        '''Forward- and back-propogate, adjusting weights via gradients
        and at magnitude via Adam algorithm.'''
        # Unpack adam params; ensure correct order
        lr, b1, b2, epsl = [
            self.adam_params[key] for key in ['lr', 'b1', 'b2', 'epsl']
        ]

        gradients = self.gradients(X, y)

        for i in range(self.w_layers):
            # Access previous moving mean & variance and replace them with current.
            self.mov_mean[i] = b1 * self.mov_mean[i] + (1-b1) * gradients[i]
            self.mov_var[i]  = b2 * self.mov_var[i]  + (1-b2) * (gradients[i]**2)
            m_hat = self.mov_mean[i] / (1 - b1**self.adam_t)
            v_hat = self.mov_var[i]  / (1 - b2**self.adam_t)

            # Update weights
            step = lr * m_hat / (np.sqrt(v_hat) + epsl)
            self.weights[i] -= step
            self.adam_t     += 1

    def loss_acc(self, y_true, y_pred, as_list=False):
        '''Compute categorical cross-entropy loss and accuracy.'''

        '''From Linear Models lecture slides, categorical cross-entropy loss
            = - sum(y_true * log(y_pred))
        For y_pred, mask zero values to avoid error from log function.
        Fill values with zero, so log(0) => 0'''
        log_mask_zero = lambda m: (
            np.ma.log(
                np.ma.array(
                    m,
                    mask = m <= 0,
                    fill_value = 0
                )
            ).data
        )
        y_pred_log = log_mask_zero(y_pred)

        classif_is_correct = lambda true, pred: int(
            np.argmax(true) == np.argmax(pred)
        )

        # Calculate loss and right/wrong for each example
        losses, correct_classifs = [], []
        for true, pred, pred_log in zip(y_true, y_pred, y_pred_log):
            losses.append(-1 * np.dot(true, pred_log))
            correct_classifs.append(classif_is_correct(true, pred))

        # Return list or average per `as_list`
        result = (losses, correct_classifs) if as_list \
            else (np.average(losses), np.average(correct_classifs))

        return result

    def predict(self, X, keep_rates=None):
        '''
        Generate predictions based on input features.

        Returns:
            One row for each example, with probabilities for each label.
            If performing dropout (i.e. if keep_rates was passed):
                 also return neuron values
                `hs` (pre-activation) and `zs` (post-activation),
                with one column for each example.
        '''

        hs = [None] * self.h_layers
        zs = [None] * self.h_layers
        # Regularize dropout by dividing activations by keep_rate,
        # i.e. activations grow inversely to keep_rate
        dropout = lambda m, keep_rate: (
            # Use binomial distribution to determine which neurons are dropped
            m / keep_rate * np.random.binomial(
                1, keep_rate, m.shape
            )
        )
        # Activate each neuron with ReLU,
        # returning max of z_hid and 0
        relu = lambda x: np.maximum(0, x)

        if keep_rates:
            X = dropout(X, keep_rates[0])

        # Pad a column of ones for multiplication with
        # the bias entry of the weight matrix
        X_p = self.pad_col(X)

        # Unactivated neuron values:
        # z_0 = (X)(w_0)
        zs[0] = X_p @ self.weights[0]
        hs[0] = relu(zs[0])

        for i in range(1, self.h_layers):
            # Unactivated output values:
            # z[i] = (h[i-1])(w[i]); pad h for bias
            h_p = self.pad_col(hs[i-1])
            if keep_rates:
                h_p = dropout(h_p, keep_rates[i])
            zs[i] = h_p @ self.weights[i]
            hs[i] = relu(zs[i])

        # Apply softmax activation to each example,
        # normalized to prevent overflow
        softmax_normalized = lambda z: (
            np.exp(z - np.amax(z)) / np.sum(np.exp(z - np.amax(z)))
        )

        z_last = self.pad_col(hs[-1]) @ self.weights[-1]

        # Activate outputs with normed softmax function from above
        y_pred = np.apply_along_axis(
            softmax_normalized,
            axis = 1,
            arr = z_last
        )
        result = (y_pred, hs, zs) if keep_rates else y_pred

        return result

    def evaluate(self, X, y, val):
        '''Calculate loss and accuracy based on predictions.
        Called between epochs to assess progress'''

        type = 'validation' if val else 'training'
        print(f'Evaluating {type} data for epoch {self.epoch_index}...')

        y_pred = self.predict(X)

        return self.loss_acc(y, y_pred)


    def eval_epoch(self, X, y, x_val, y_val, start_time):
        '''Calculate loss and accuracy for training and validation sets;
        append epoch-wise stats for later saving and progress printing.'''

        # Current epoch performance stats
        curr_loss_train, curr_acc_train = self.evaluate(X,     y,     val=False)
        curr_loss_val,   curr_acc_val   = self.evaluate(x_val, y_val, val=True)
        curr_elapsed                    = np.round(time.time() - start_time, 2)
        curr_performance = dict(
            loss_train = curr_loss_train,
            loss_val   = curr_loss_val,
            acc_train  = curr_acc_train,
            acc_val    = curr_acc_val,
            total_sec_elapsed = curr_elapsed
        )
        assert None not in [curr_performance.values()], 'Performance values incomplete'

        # Append to df of epoch-wise performance stats
        self.epoch_performance = self.df_add_dict(
            self.epoch_performance, curr_performance, self.epoch_index
        )

        # Calculate values at the percentiles of weights for each layer
        # Only accessed if verbosity >= 2
        weight_stats_keys = ['all', '99p', '95p', '50p']
        percentiles = [100, 99, 95, 50]
        curr_ws = {
            key: ([None] * self.w_layers) for key in weight_stats_keys
        }
        for i in range(self.w_layers):
            for key, percentile in zip(weight_stats_keys, percentiles):
                curr_ws[key][i] = self.middle_percentile_and_range(
                    self.weights[i], percentile
                )
        # Replace weight stats with current epoch's weight stats
        self.epoch_weight_stats = pd.DataFrame(
            curr_ws, index=[i for i in range(self.w_layers)]
        ).style.set_properties(**{'white-space': 'pre'})


    def gradients(self, X, y_true):
        '''Forward- and backpropogate, returning
        gradients of loss with respect to weights
        for each weight layer.'''

        # Reshuffle weights according to markdown above
        if self.keep_rates_reshuffle:
            if self.keep_rates_range:
                min, max = self.keep_rates_range
                self.keep_rates = [np.random.uniform(min, max) for _ in range(self.w_layers)]
            if self.keep_rates_manual:
                np.random.shuffle(self.keep_rates_manual)

        y_pred, hs, zs = self.predict(X, self.keep_rates)

        gradients = [None] * self.w_layers

        # For softmax output activation,
        # dL/dw_last = (y_pred - y_true)(h)
        dL_dz = y_pred - y_true
        gradients[-1] = self.pad_col(hs[-1]).T @ dL_dz

        relu_grad = lambda m: m > 0

        # Backpropogation bewteen hidden layers
        for i in np.arange(self.w_layers-2, -1, -1):
            w_no_bias = self.weights[i+1][:-1,:]
            dL_dz = (dL_dz @ w_no_bias.T) * relu_grad(zs[i])
            # If reached last backprop (to input layer)
            prev_activations = X if i==0 else hs[i-1]
            gradients[i] = self.pad_col(prev_activations).T @ dL_dz

        # Average gradients by dividing by batch size
        return [g / X.shape[0] for g in gradients]