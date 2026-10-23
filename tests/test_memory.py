"""Tests for memory.py: SessionStore facts, relevance-ranked search,
correction handling, secret filtering, and session file safety."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import memory as memory_mod
from memory import SessionStore


def fresh_store():
    return SessionStore(folder=tempfile.mkdtemp())


class TestSearch(unittest.TestCase):
    def test_most_relevant_fact_first(self):
        s = fresh_store()
        s.add_facts([
            "Joey works at Walmart as an hourly employee",
            "Joey likes pepperoni pizza on Fridays",
        ])
        out = s.search("joey works at walmart", limit=2)
        self.assertEqual(out[0], "Joey works at Walmart as an hourly employee")

    def test_falls_back_to_newest_when_nothing_overlaps(self):
        s = fresh_store()
        for i in range(4):
            s.add_facts([f"rivers {i}"])
        out = s.search("zzz qqq")
        # min(limit=8, 3) newest facts, newest first
        self.assertEqual(out, ["rivers 3", "rivers 2", "rivers 1"])

    def test_empty_query_returns_nothing(self):
        s = fresh_store()
        s.add_facts(["Joey works at Walmart as an hourly employee"])
        self.assertEqual(s.search(""), [])
        self.assertEqual(s.search(None), [])
        self.assertEqual(s.search("!!!"), [])

    def test_no_facts_returns_nothing(self):
        self.assertEqual(fresh_store().search("anything"), [])


class TestAddFacts(unittest.TestCase):
    def test_exact_duplicate_not_readded(self):
        s = fresh_store()
        s.add_facts(["Joey works at Walmart"])
        self.assertEqual(s.add_facts(["  Joey   works at Walmart "]), [])
        self.assertEqual(len(s.facts), 1)

    def test_correction_replaces_old_fact(self):
        s = fresh_store()
        s.add_facts(["Joey's shift is 7am to 3pm"])
        added = s.add_facts(["Joey works the 7am to 3pm shift"])
        self.assertEqual(added, ["Joey works the 7am to 3pm shift"])
        self.assertEqual(len(s.facts), 1)
        self.assertEqual(s.facts[0]["text"], "Joey works the 7am to 3pm shift")
        self.assertEqual(s.facts[0]["replaces"], "Joey's shift is 7am to 3pm")

    def test_unrelated_fact_is_appended(self):
        s = fresh_store()
        s.add_facts(["Joey works at Walmart"])
        s.add_facts(["Immich runs on the homelab server"])
        self.assertEqual(len(s.facts), 2)

    def test_secret_looking_facts_rejected(self):
        s = fresh_store()
        added = s.add_facts([
            "My password is hunter2 secret",
            "The api key is abc123",
            "Card number 4111111111111111 on file",
            "Joey starts work Monday morning",
        ])
        self.assertEqual(added, ["Joey starts work Monday morning"])
        self.assertEqual(len(s.facts), 1)

    def test_length_limits(self):
        s = fresh_store()
        self.assertEqual(s.add_facts(["ab"]), [])            # too short
        self.assertEqual(s.add_facts(["x" * 201]), [])       # too long
        self.assertEqual(len(s.add_facts(["x" * 200])), 1)   # boundary ok

    def test_non_strings_skipped(self):
        s = fresh_store()
        self.assertEqual(s.add_facts([123, None, "valid fact here ok"]), ["valid fact here ok"])

    def test_max_facts_drops_oldest(self):
        s = fresh_store()
        old = memory_mod.MAX_FACTS
        memory_mod.MAX_FACTS = 5
        try:
            import uuid
            for _ in range(7):
                h = uuid.uuid4().hex[:8]
                s.add_facts([f"delta {h} tango {uuid.uuid4().hex[:8]}"])
            self.assertEqual(len(s.facts), 5)
        finally:
            memory_mod.MAX_FACTS = old

    def test_limit_caps_added(self):
        s = fresh_store()
        added = s.add_facts(
            ["alpha orchard kites", "zulu nine lanterns", "mike five harbor"],
            limit=2,
        )
        self.assertEqual(len(added), 2)


class TestFactsAccess(unittest.TestCase):
    def test_facts_returns_a_copy(self):
        s = fresh_store()
        s.add_facts(["Joey works at Walmart"])
        s.facts.append({"text": "poison"})
        self.assertEqual(len(s.facts), 1)

    def test_remove_by_index(self):
        s = fresh_store()
        s.add_facts(["Joey works at Walmart", "Immich runs on the homelab server"])
        self.assertEqual(s.remove(1), "Joey works at Walmart")
        self.assertIsNone(s.remove(99))
        self.assertEqual(len(s.facts), 1)

    def test_clear(self):
        s = fresh_store()
        s.add_facts(["Joey works at Walmart"])
        s.clear()
        self.assertEqual(s.facts, [])

    def test_facts_survive_reload(self):
        tmp = tempfile.mkdtemp()
        SessionStore(folder=tmp).add_facts(["Joey works at Walmart"])
        self.assertEqual(SessionStore(folder=tmp).memory_texts(), ["Joey works at Walmart"])

    def test_corrupt_memory_file_recovers(self):
        tmp = Path(tempfile.mkdtemp())
        (tmp / "memory.json").write_text("{not valid json", encoding="utf-8")
        s = SessionStore(folder=str(tmp))
        self.assertEqual(s.facts, [])
        self.assertIsNotNone(s.warning)
        self.assertEqual(len(list(tmp.glob("memory.corrupt-*.json"))), 1)
        # usable afterwards
        s.add_facts(["Joey works at Walmart"])
        self.assertEqual(len(s.facts), 1)


class TestSessions(unittest.TestCase):
    def test_save_and_load_roundtrip(self):
        s = fresh_store()
        msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
        p = s.save_session("20261023_112000", "2026-10-23", "llama3.2", msgs)
        self.assertTrue(p.is_file())
        data = s.load_session("20261023_112000")
        self.assertEqual(data["messages"], msgs)

    def test_load_session_rejects_path_traversal(self):
        s = fresh_store()
        self.assertIsNone(s.load_session("../memory"))
        self.assertIsNone(s.load_session("../../etc/passwd"))
        self.assertIsNone(s.load_session("nope;rm -rf"))

    def test_load_session_missing_returns_none(self):
        self.assertIsNone(fresh_store().load_session("does_not_exist"))

    def test_list_sessions_skips_corrupt_files(self):
        s = fresh_store()
        s.save_session("good_one", "2026-10-23", "llama3.2", [{"role": "user", "content": "hi"}])
        (s.dir / "session_broken.json").write_text("{oops", encoding="utf-8")
        listed = s.list_sessions()
        self.assertEqual([e["id"] for e in listed], ["good_one"])
        self.assertEqual(listed[0]["preview"], "hi")
        self.assertEqual(listed[0]["message_count"], 1)


if __name__ == "__main__":
    unittest.main()
