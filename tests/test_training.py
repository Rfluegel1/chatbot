import unittest
import warnings

from src.training import read_json


class TestReadFunction(unittest.TestCase):

    def test_read(self):
        filename = 'tests/valid.json'
        expected = {'key': 'value'}
        self.assertEqual(read_json(filename), expected)

    def test_file_close(self):
        filename = 'tests/valid.json'

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            read_json(filename)

            for warning in w:
                self.assertNotEqual(warning.category, ResourceWarning)
