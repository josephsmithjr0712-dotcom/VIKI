"""Tests for v0.3.9: multi-word chat search matches all words, any order."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chats import ChatStore


class TestMultiWordSearch(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="viki-chat-search-test-")
        self.store = ChatStore(folder=self.tmp)
        t = self.store.create(title="Work", kind="side")
        t["messages"] = [
            {"role": "user", "content": "My onboarding is Monday, start date to follow"},
            {"role": "user", "content": "Buy milk and eggs"},
        ]
        self.store.save(t)
        self.thread_id = t["id"]

    def test_words_out_of_order_match(self):
        hits = self.store.search("onboarding date")
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["chat_id"], self.thread_id)
        self.assertIn("onboarding", hits[0]["snippet"].lower())

    def test_words_in_reverse_order_match(self):
        hits = self.store.search("date onboarding")
        self.assertEqual(len(hits), 1)

    def test_missing_one_word_does_not_match(self):
        self.assertEqual(self.store.search("onboarding pizza"), [])

    def test_exact_phrase_still_matches(self):
        hits = self.store.search("start date")
        self.assertEqual(len(hits), 1)

    def test_single_word_unchanged(self):
        hits = self.store.search("milk")
        self.assertEqual(len(hits), 1)
        self.assertEqual(self.store.search(""), [])


if __name__ == "__main__":
    unittest.main()
