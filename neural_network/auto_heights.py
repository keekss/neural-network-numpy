from typing import List, Tuple
import numpy as np
import logging

def auto_heights(
        h_layers: int = 3,  # Number of hidden layers
        shape: str = 'flat',
        max_height: int = 200,
        shrink_factor: float = 0.5
) -> Tuple[List[int], str]:
    """
    Generate a list of hidden layer heights based on the specified shape.

    This method calculates the heights of the hidden layers based on the specified shape,
    maximum height, and shrink factor.

    Args:
        h_layers (int): Number of hidden layers.
        shape (str): Shape of the hidden layers. Must be one of: 'flat', 'contracting', 'expanding'.
        max_height (int): Maximum height of the hidden layers.
        shrink_factor (float): Shrink factor to determine the minimum height of the hidden layers.

    Returns:
        Tuple[List[int], str]: Tuple containing the list of hidden layer heights and a description of the heights.

    Raises:
        ValueError: If a height of zero is encountered or an invalid shape is provided.

    Warnings:
        If `max_height` is provided as a float, it will be rounded to the nearest integer.

    """

    # Ensure shrink_factor is between 0 and 1, defaulting to flat shape if not
    if not (0 < shrink_factor <= 1):
        logging.warning(
            f'Invalid shrink factor: {shrink_factor}. Must be between 0 and 1. Defaulting to flat shape.'
        )
        shape = 'flat'

    # Ensure max_height is an integer, rounding if not
    if not isinstance(max_height, int):
        old_max_height = max_height
        max_height = round(max_height)
        logging.warning(
            f'max_height was provided as a float ({old_max_height}). It has been rounded to the nearest integer ({max_height}).'
        )

    # Calculate the heights of the hidden layers based on the specified shape
    height_min, height_max = max_height * shrink_factor, max_height
    left_right_by_shape = {
        'flat': (height_max, height_max),
        'contracting': (height_max, height_min),
        'expanding': (height_min, height_max),
    }
    try:
        left, right = left_right_by_shape[shape]
    except KeyError:
        logging.warning(
            f'Invalid shape. Must be one of: {list(left_right_by_shape.keys())}. Defaulting to flat shape.'
        )
        left, right = left_right_by_shape['flat']

    # Generate the heights of the hidden layers
    heights = np.linspace(left, right, num=h_layers, dtype='int')
    # Ensure no heights of zero
    if 0 in heights:
        raise ValueError('Height of zero not allowed')

    return heights