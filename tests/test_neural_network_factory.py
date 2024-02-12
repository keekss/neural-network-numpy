import unittest
from modules.factory import NeuralNetworkFactory, Container
import yaml
import numpy as np

class TestNeuralNetworkFactory(unittest.TestCase):
    def setUp(self) -> None:
                # Load the configuration from the YAML file
        with open('config.yml', 'r') as f:
            config_data = yaml.safe_load(f)

        # Initialize the container and update the configuration
        self.container = Container(config = config_data)



    def test_create(self):
        
        # Create a neural network with the configuration
        self.obj = NeuralNetworkFactory.create(**self.container.config.neural_network())

        self.assertTrue(np.array_equal(
            self.obj.heights,
            [200,200,200]
        ))

    def test_auto_heights(self):

        test_cases = [
            ('flat', [200, 200, 200]),
            ('contracting', [200, 150, 100]),
            ('expanding', [100, 150, 200])
        ]

        print(type(self.container.config.neural_network()))

        for shape, expected in test_cases:
            with self.subTest(shape=shape):
                # Update the configuration with the new shape
                self.container.config.neural_network().update({'shape': shape})


                # Create a neural network with the updated configuration
                neural_network = NeuralNetworkFactory.create(**self.container.config.neural_network())

                print(neural_network.heights, expected)

                # Check that the heights of the neural network match the expected heights
                self.assertTrue(np.array_equal(neural_network.heights, expected))






if __name__ == '__main__':
    unittest.main()