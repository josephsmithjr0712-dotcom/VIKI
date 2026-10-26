"""Tests for VIKI's user profile (v0.3.0)."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from profile import UserProfile


class TestUserProfile(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="viki-profile-test-")
        self.p = UserProfile(folder=self.tmp)

    def test_add_dedup_remove(self):
        self.assertTrue(self.p.add("like", "black coffee"))
        self.assertFalse(self.p.add("like", "black coffee"))  # duplicate
        self.assertIn("black coffee", self.p.as_dict()["likes"])
        self.assertTrue(self.p.remove("like", "black coffee"))
        self.assertFalse(self.p.remove("like", "black coffee"))
        self.assertEqual(self.p.as_dict()["likes"], [])

    def test_favorites_keyed(self):
        self.assertTrue(self.p.add("favorite", "black, no sugar", favorite_key="coffee"))
        self.assertFalse(self.p.add("favorite", "oat milk", favorite_key=""))  # no key
        self.assertEqual(self.p.as_dict()["favorites"], {"coffee": "black, no sugar"})
        self.assertTrue(self.p.remove("favorite", "coffee"))

    def test_invalid_kinds_and_values_rejected(self):
        self.assertFalse(self.p.add("mood", "happy"))
        self.assertFalse(self.p.add("like", "x"))  # too short
        self.assertFalse(self.p.add("like", "my password is hunter2"))  # secret filter
        self.assertFalse(self.p.add("like", "x" * 200))  # too long
        self.assertEqual(self.p.text(), "")

    def test_text_block_omits_empty_sections(self):
        self.p.add("like", "quiet mornings")
        self.p.add("hobby", "fishing")
        text = self.p.text()
        self.assertIn("Likes: quiet mornings", text)
        self.assertIn("Hobbies: fishing", text)
        self.assertNotIn("Dislikes", text)

    def test_persists_across_instances(self):
        self.p.add("dislike", "loud alarms")
        other = UserProfile(folder=self.tmp)
        self.assertIn("loud alarms", other.as_dict()["dislikes"])

    def test_import_from_facts(self):
        added = self.p.import_from_facts([
            "User likes black coffee.",
            "User works at Walmart.",  # not a preference: skipped
            "User hates loud alarms.",
            "User likes black coffee.",  # duplicate: skipped on second pass
        ])
        kinds = [k for k, _ in added]
        self.assertIn("like", kinds)
        self.assertIn("dislike", kinds)
        self.assertEqual(len(added), 2)
        self.assertIn("black coffee", self.p.text())

    def test_corrupt_file_starts_fresh(self):
        Path(self.tmp, "profile.json").write_text("{not json", encoding="utf-8")
        p = UserProfile(folder=self.tmp)
        self.assertEqual(p.as_dict()["likes"], [])


if __name__ == "__main__":
    unittest.main()
