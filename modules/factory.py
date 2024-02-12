# Standard library imports
from enum import Enum
from typing import List, Dict, Union, Optional
import logging

# Third-party imports
import numpy as np
from dependency_injector import providers, containers

# Local application/library specific imports
import scripts.logging_config as _
from NeuralNetwork import NeuralNetwork

class Shape(Enum):
    """
    An enumeration representing the shape of the hidden layers in a neural network.
    
    The number of neurons in the hidden layers ______ from the input layer to the output layer.

    Attributes:
        FLAT:        is constant
        CONTRACTING: decreases
        EXPANDING:   increases
    """
    FLAT = 'flat'
    CONTRACTING = 'contracting'
    EXPANDING = 'expanding'

    @staticmethod
    def from_string(shape_string: str):
        if isinstance(shape_string, Shape):
            return shape_string
        try:
            return Shape[shape_string.upper()]
        except KeyError:
            error_message = f"Invalid shape string '{shape_string}'. Must be one of: {list(Shape._member_names_)}"
            logging.error(error_message)
            raise ValueError(error_message)

class NeuralNetworkFactory:
    @staticmethod
    def create(
            inputs: int,
            outputs: int,
            **kwargs
    ) -> NeuralNetwork:
        """
        Creates a new instance of the NeuralNetwork class and initializes it with the given parameters.

        :param inputs: The number of inputs to the neural network.
        :param outputs: The number of outputs from the neural network.
        :param manual_heights: A list of heights for each layer in the neural network. If this is None, the heights will be automatically calculated.
        :param auto_heights_params: A dictionary of parameters for automatically calculating the heights of the layers. This is ignored if manual_heights is not None.
        :param kwargs: Additional keyword arguments to pass to the NeuralNetwork constructor.
        :return: A new instance of the NeuralNetwork class.
        """
        obj = NeuralNetwork(inputs = inputs, outputs = outputs)
        obj = NeuralNetworkFactory.initialize_heights(
            obj = obj,
            **kwargs
        )
        obj = NeuralNetworkFactory.initialize_weights(obj = obj)

        obj = NeuralNetworkFactory.initialize_epoch_variables(obj = obj)

        obj = NeuralNetworkFactory.initialize_adam_params(obj = obj)
        
        return obj

    @staticmethod
    def initialize_heights(
            obj: NeuralNetwork,
            manual_heights: Optional[List[int]] = None,
            **kwargs
    ) -> NeuralNetwork:
        """
        Initializes the heights of the neural network.

        Args:
            obj (NeuralNetwork): The neural network object.
            manual_heights (Optional[List[int]], optional): A list of manual heights for each layer. Defaults to None.
            auto_heights_params (Optional[Dict[str, Union[int, float, str]]], optional): Parameters for automatic height calculation. Defaults to None.

        Returns:
            NeuralNetwork: The neural network object with initialized heights.
        """
        if manual_heights:
            obj.heights = manual_heights
            obj.heights_shape = 'manual'
            obj.heights_descrip = f'manual: {manual_heights}'
        else:
            obj.heights = NeuralNetworkFactory.auto_heights(**kwargs)
            obj.heights_shape = kwargs.get('shape', 'flat')
            obj.heights_descrip = f'{obj.heights_shape}, ({obj.heights[0]} => {int(obj.heights[-1])})'
        obj.h_layers = len(obj.heights)
        obj.w_layers = obj.h_layers + 1
        return obj

    @staticmethod
    def initialize_weights(obj: NeuralNetwork) -> NeuralNetwork:
        obj.weights = [None] * obj.w_layers
        obj.weights[0] = np.random.normal(
            loc=0,
            scale=np.sqrt(2/obj.inputs),
            size=(obj.inputs + 1, obj.heights[0])
        )
        for i in range(1, obj.h_layers):
            obj.weights[i] = np.random.normal(
                0, np.sqrt(2/obj.inputs),
                size=(obj.heights[i-1]+1, obj.heights[i])
            )
        obj.weights[-1] = np.random.normal(
            0, np.sqrt(2/obj.inputs),
            size=(obj.heights[-1]+1, obj.outputs)
        )
        return obj

    @staticmethod
    def initialize_epoch_variables(obj: NeuralNetwork) -> NeuralNetwork:
        obj.epoch_performance = None
        obj.epoch_weights = []
        obj.epoch_weight_stats = []
        return obj

    @staticmethod
    def initialize_adam_params(obj: NeuralNetwork) -> NeuralNetwork:
        obj.adam_params = {
            'lr': 3e-4,
            'b1': 1-1e-1,
            'b2': 1-1e-3,
            'epsl': 1e-8,
        }
        obj.adam_t = 1
        obj.mov_mean = [0] * obj.w_layers
        obj.mov_var = [0] * obj.w_layers
        return obj


    
    @staticmethod
    def auto_heights(
            h_layers: int = 3,  # Number of hidden layers
            shape: Shape = Shape.FLAT,
            max_height: int = 200,
            shrink_factor: float = 0.5,
            **kwargs
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
        left, right = left_right_by_shape[Shape.from_string(shape)]

        # Generate the heights of the hidden layers
        heights = np.linspace(left, right, num=h_layers, dtype='int')
        # Ensure no heights of zero
        if 0 in heights:
            error_message = 'Height of zero not allowed'
            logging.error(error_message)
            raise ValueError(error_message)

        return heights
    
class Container(containers.DeclarativeContainer):
    """
    IoC container for dependency injection.

    This container manages the dependencies and configurations for the application. It provides a centralized location for creating and accessing instances of classes and configurations.

    Attributes:
        config (providers.Configuration): A provider for accessing the application's configuration settings.
        neural_network_factory (providers.Factory): A factory provider for creating and managing instances of the NeuralNetworkFactory class.
    """
    config = providers.Configuration('config')
    neural_network_factory = providers.Factory(NeuralNetworkFactory)