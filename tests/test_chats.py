"""Tests for VIKI chat threads (v0.3.0): main chat + side chats."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chats import ChatStore, context_history


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

    def test_main_chat_protected_from_delete(self):
        main = self.store.ensure_main()
        self.assertFalse(self.store.delete(main["id"], protect_main=True))
        self.assertIsNotNone(self.store.get(main["id"]))
        side = self.store.create(title="Side", kind="side")
        self.assertTrue(self.store.delete(side["id"], protect_main=True))

    def test_search_finds_matches_across_threads(self):
        t1 = self.store.create(title="Trip", kind="side")
        t1["messages"] = [{"role": "user", "content": "Book a hotel in Killbuck"},
                         {"role": "agent", "content": "Done, booked it"}]
        self.store.save(t1)
        t2 = self.store.create(title="Groceries", kind="side")
        t2["messages"] = [{"role": "user", "content": "Buy milk in Killbuck"}]
        self.store.save(t2)
        hits = self.store.search("killbuck")
        self.assertEqual(len(hits), 2)
        self.assertEqual({h["chat_id"] for h in hits}, {t1["id"], t2["id"]})
        self.assertIn("Killbuck", hits[0]["snippet"])
        self.assertEqual(self.store.search(""), [])
        self.assertEqual(self.store.search("zzz-no-such-word"), [])

    def test_side_chats_do_not_share_messages(self):
        a = self.store.create(title="A", kind="side")
        b = self.store.create(title="B", kind="side")
        a["messages"] = [{"role": "user", "content": "only in A"}]
        self.store.save(a)
        self.assertEqual(self.store.get(b["id"])["messages"], [])


if __name__ == "__main__":
    unittest.main()


class TestContextHistory(unittest.TestCase):
    """v0.3.8: long threads compress older messages into one summary."""

    def _msgs(self, n):
        out = []
        for i in range(n):
            role = "user" if i % 2 == 0 else "assistant"
            out.append({"role": role, "content": f"msg {i}", "time": "t",
                        "extra": "stripped"})
        return out

    def test_short_thread_passes_through(self):
        msgs = self._msgs(5)
        hist = context_history(msgs)
        self.assertEqual(len(hist), 5)
        self.assertTrue(all(set(m) == {"role", "content"} for m in hist))
        self.assertEqual(hist[0]["content"], "msg 0")

    def test_long_thread_summarizes_older_keeps_recent(self):
        msgs = self._msgs(25)
        hist = context_history(msgs)  # default keep=20
        self.assertEqual(len(hist), 21)
        self.assertEqual(hist[0]["role"], "system")
        self.assertIn("5 earlier messages", hist[0]["content"])
        self.assertIn("msg 0", hist[0]["content"])  # older user topic kept
        self.assertEqual([m["content"] for m in hist[1:]],
                         [f"msg {i}" for i in range(5, 25)])
        self.assertNotIn("msg 1", [m["content"] for m in hist[1:]])

    def test_summary_dedupes_and_caps_topics(self):
        msgs = [{"role": "user", "content": "same topic"}] * 12 + self._msgs(20)
        hist = context_history(msgs)
        self.assertEqual(hist[0]["content"].count("same topic"), 1)

    def test_no_user_messages_in_older_part(self):
        msgs = [{"role": "assistant", "content": "a"}] * 5 + self._msgs(20)
        hist = context_history(msgs)
        self.assertIn("5 earlier messages", hist[0]["content"])
        self.assertEqual(len(hist), 21)

    def test_custom_keep(self):
        hist = context_history(self._msgs(10), keep=3)
        self.assertEqual(len(hist), 4)
        self.assertEqual([m["content"] for m in hist[1:]], ["msg 7", "msg 8", "msg 9"])
