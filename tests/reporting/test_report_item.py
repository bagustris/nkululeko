# test_report_item.py - unit tests for nkululeko/reporting/report_item.py
import os
import tempfile
import unittest

from nkululeko.reporting.report_item import ReportItem


class TestReportItem(unittest.TestCase):
    def test_init_without_image(self):
        """Test ReportItem initialization without image."""
        item = ReportItem("topic", "caption", "contents")
        self.assertEqual(item.topic, "topic")
        self.assertEqual(item.caption, "caption")
        self.assertEqual(item.contents, "contents")
        self.assertFalse(item.has_image)

    def test_init_with_image(self):
        """Test ReportItem initialization with image."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            image_path = f.name
        try:
            item = ReportItem("topic", "caption", "contents", image=image_path)
            self.assertEqual(item.topic, "topic")
            self.assertEqual(item.caption, "caption")
            self.assertEqual(item.contents, "contents")
            self.assertTrue(item.has_image)
            self.assertTrue(os.path.isabs(item.image))
        finally:
            os.unlink(image_path)

    def test_to_string(self):
        """Test to_string method."""
        item = ReportItem("test_topic", "test_caption", "test_contents")
        result = item.to_string()
        self.assertIn("test_topic", result)
        self.assertIn("test_caption", result)
        self.assertIn("test_contents", result)


if __name__ == "__main__":
    unittest.main()
