"""Tests for VIKI chat threads (v0.3.0): main chat + side chats."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chats import ChatStore


class TestChatStore(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="viki-chats-test-")
        self.store = ChatStore(folder=self.tmp)

    def test_create_list_get_roundtrip(self):
        t = self.store.create(title="Groceries", kind="side")
        self.assertEqual(t["kind"], "side")
        self.assertEqual(t["messages"], [])
        listed = self.store.list_threads()
        self.assertEqual(len(listed), 1)
        self.assertEqual(listed[0]["title"], "Groceries")
        self.assertNotIn("messages", listed[0])  # summaries carry no bodies
        got = self.store.get(t["id"])
        self.assertEqual(got["title"], "Groceries")

    def test_save_autotitles_from_first_user_message(self):
        t = self.store.create(kind="side")
        t["messages"] = [{"role": "user", "content": "Plan a trip to Killbuck next weekend please"}]
        saved = self.store.save(t)
        self.assertEqual(saved["title"], "Plan a trip to Killbuck next weekend please")

    def test_save_persists_across_instances(self):
        t = self.store.create(title="Side quest", kind="side")
        t["messages"] = [{"role": "user", "content": "hi"}]
        self.store.save(t)
        other = ChatStore(folder=self.tmp)
        self.assertEqual(other.get(t["id"])["messages"][0]["content"], "hi")

    def test_rename_and_delete(self):
        t = self.store.create(title="Old", kind="side")
        renamed = self.store.rename(t["id"], "New name")
        self.assertEqual(renamed["title"], "New name")
        self.assertIsNone(self.store.rename("missing-id", "x"))
        self.assertTrue(self.store.delete(t["id"]))
        self.assertIsNone(self.store.get(t["id"]))
        self.assertFalse(self.store.delete(t["id"]))

    def test_unsafe_ids_rejected(self):
        self.assertIsNone(self.store.get("../../etc/passwd"))
        self.assertIsNone(self.store.get(""))
        self.assertFalse(self.store.delete("../escape"))

    def test_ensure_main_creates_once(self):
        main = self.store.ensure_main()
        self.assertEqual(main["kind"], "main")
        self.assertEqual(main["title"], "Main chat")
        again = self.store.ensure_main()
        self.assertEqual(again["id"], main["id"])
        self.assertEqual(len(self.store.list_threads()), 1)

    def test_side_chats_do_not_share_messages(self):
        a = self.store.create(title="A", kind="side")
        b = self.store.create(title="B", kind="side")
        a["messages"] = [{"role": "user", "content": "only in A"}]
        self.store.save(a)
        self.assertEqual(self.store.get(b["id"])["messages"], [])


if __name__ == "__main__":
    unittest.main()
