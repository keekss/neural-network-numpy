from typing import List, Dict, Union, Optional

import numpy as np
from enum import Enum

import logging
import scripts.logging_config as _

class Shape(Enum):
    FLAT = 'flat'
    CONTRACTING = 'contracting'
    EXPANDING = 'expanding'

from NeuralNetwork import NeuralNetwork

class NeuralNetworkFactory:
    @staticmethod
    def initialize_heights(manual_heights, auto_heights_params):
        if manual_heights:
            heights = manual_heights
            heights_shape = 'manual'
            heights_descrip = f'manual: {manual_heights}'
        else:
            if auto_heights_params is None:
                auto_heights_params = {}
            heights = NeuralNetworkFactory.auto_heights(**auto_heights_params)
            heights_shape = auto_heights_params.get('shape', 'flat')
            heights_descrip = f'{heights_shape}, ({heights[0]} => {int(heights[-1])})'
        h_layers = len(heights)
        w_layers = h_layers + 1
        return heights, heights_shape, heights_descrip, h_layers, w_layers

    @staticmethod
    def initialize_weights(inputs: int, outputs: int):
        weights = [None] * w_layers
        weights[0] = np.random.normal(
            loc=0,
            scale=np.sqrt(2/inputs),
            size=(inputs + 1, heights[0])
        )
        for i in range(1, h_layers):
            weights[i] = np.random.normal(
                0, np.sqrt(2/inputs),
                size=(heights[i-1]+1, heights[i])
            )
        weights[-1] = np.random.normal(
            0, np.sqrt(2/inputs),
            size=(heights[-1]+1, outputs)
        )

    @staticmethod
    def initialize_epoch_variables():
        # Copy the logic from the NeuralNetwork class
        # ...

    @staticmethod
    def initialize_adam_params():
        # Copy the logic from the NeuralNetwork class
        # ...

    @staticmethod
    def create(
            inputs: int,
            outputs: int,
            manual_heights: Optional[List[int]] = None,
            auto_heights_params: Optional[Dict[str, Union[int, float, str]]] = None,
    ) -> NeuralNetwork:
        obj = NeuralNetwork()
        obj.heights, obj.heights_shape, obj.heights_descrip, obj.h_layers, obj.w_layers = NeuralNetworkFactory.initialize_heights(manual_heights, auto_heights_params)
        obj.weights = NeuralNetworkFactory.initialize_weights(inputs, outputs)
        obj.epoch_performance, obj.epoch_weights, obj.epoch_weight_stats = NeuralNetworkFactory.initialize_epoch_variables()
        obj.adam_params, obj.adam_t, obj.mov_mean, obj.mov_var = NeuralNetworkFactory.initialize_adam_params()
        return obj
    
    @staticmethod
    def auto_heights(
            h_layers: int = 3,  # Number of hidden layers
            shape: Shape = Shape.FLAT,
            max_height: int = 200,
            shrink_factor: float = 0.5
    ) -> List[int]:
        """
        @brief Generate a list of hidden layer heights based on the specified shape.

        This method calculates the heights of the hidden layers based on the specified shape,
        maximum height, and shrink factor. If the shrink factor is not between 0 and 1, an error is logged and an exception is raised. If max_height is provided as a float, it is rounded to the nearest integer and a warning is logged.

        @param h_layers Number of hidden layers.
        @param shape Shape of the hidden layers. Must be one of: 'flat', 'contracting', 'expanding'.
        @param max_height Maximum height of the hidden layers. If provided as a float, it is rounded to the nearest integer.
        @param shrink_factor Shrink factor to determine the minimum height of the hidden layers. Must be between 0 and 1.

        @return The list of hidden layer heights.

        @throws ValueError If a height of zero is encountered, an invalid shape is provided, or the shrink factor is not between 0 and 1.
        """

        # Ensure shrink_factor is between 0 and 1, defaulting to flat shape if not
        if not (0 < shrink_factor <= 1):
            error_message = f'Invalid shrink factor: {shrink_factor}. Must be between 0 and 1.'
            logging.error(error_message)
            raise ValueError(error_message)

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
            Shape.FLAT:        (height_max, height_max),
            Shape.CONTRACTING: (height_max, height_min),
            Shape.EXPANDING:   (height_min, height_max),
        }
        try:
            left, right = left_right_by_shape[shape]
        except KeyError:
            error_message = f'Invalid shape. Must be one of: {list(Shape)}.'
            logging.error(error_message)
            raise ValueError(error_message)

        # Generate the heights of the hidden layers
        heights = np.linspace(left, right, num=h_layers, dtype='int')
        # Ensure no heights of zero
        if 0 in heights:
            error_message = 'Height of zero not allowed'
            logging.error(error_message)
            raise ValueError(error_message)

        return heights