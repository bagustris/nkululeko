# test_defines.py - unit tests for nkululeko/reporting/defines.py
import unittest

from nkululeko.reporting.defines import Header


class TestDefineBase(unittest.TestCase):
    def test_attribute_values(self):
        """Test that _attribute_values returns class attributes."""
        values = Header._attribute_values()
        self.assertIn("Results", values)
        self.assertIn("Data exploration", values)

    def test_assert_has_attribute_value_valid(self):
        """Test that valid values don't raise exceptions."""
        Header._assert_has_attribute_value("Results")
        Header._assert_has_attribute_value("Data exploration")

    def test_assert_has_attribute_value_invalid(self):
        """Test that invalid values raise ValueError."""
        with self.assertRaises(ValueError) as context:
            Header._assert_has_attribute_value("InvalidValue")
        self.assertIn("Invalid value", str(context.exception))
        self.assertIn("Valid values", str(context.exception))


if __name__ == "__main__":
    unittest.main()
