import unittest
import pandas as pd


class MyTestCase(unittest.TestCase):
    def test_gini4(self):
         self.assertEqual(5,0.375)


if __name__ == '__main__':
    unittest.main()


