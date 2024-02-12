import unittest
import numpy as np

from neural_network.auto_heights import shaped_heights

class TestShapedHeights(unittest.TestCase):
    def test_shaped_heights_logic(self):
        test_cases = [
            ('flat', lambda heights: all(height == 200 for height in heights)),
            ('contracting', lambda heights: all(h1 >= h2 for h1, h2 in zip(heights, heights[1:]))),
            ('expanding', lambda heights: all(h1 <= h2 for h1, h2 in zip(heights, heights[1:])))
        ]

        for shape, condition in test_cases:
            with self.subTest(shape=shape):
                heights = shaped_heights(shape=shape)
                self.assertTrue(condition(heights))

        # Test for invalid shape
        with self.subTest(shape='invalid'):
            heights = shaped_heights(shape='invalid')
            self.assertTrue(np.array_equal(heights, np.array([200, 200, 200])))  # same condition as 'flat'

if __name__ == '__main__':
    unittest.main()