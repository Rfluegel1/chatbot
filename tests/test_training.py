import unittest
from src.training import add


class TestAddFunction(unittest.TestCase):

    def test_add(self):
        self.assertEqual(add(1, 2), 3)

    def test_add_negative(self):
        self.assertEqual(add(-1, 1), 0)