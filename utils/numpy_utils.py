import numpy as np

def shuffle_Xy(X ,y):
    '''Reshuffle X and y by randomly permuting their indices.  Maintain shape.'''
    shuffled_indices = np.random.permutation(X.shape[0])
    return X[shuffled_indices], y[shuffled_indices]


def pad_col(m):
    '''Pad a column of ones to a numpy array.
    Used for calculations involving biases.'''
    ones_col = np.ones(m.shape[0], 1)
    return np.hstack([m, ones_col])


def middle_percentile_and_range(arr, percentile):
    '''Percentile strings used for weight distribution analysis.'''
    min, max = np.percentile(arr, [50-percentile/2, 50+percentile/2])
    range = max - min
    return f'({range:<8.4f}: {min:<8.4f} => {max:>8.4f})'

