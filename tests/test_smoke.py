"""Smoke tests for VIKI core: agent loop, tools, memory. Ollama is mocked."""
import json
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import ollama
except ImportError:
    # Hourly CI sandbox has no ollama package; the client is mocked anyway,
    # so register a minimal stub before agent.py is imported (agent.py does
    # a plain `import ollama`, which will pick this up from sys.modules).
    import types

    ollama = types.ModuleType("ollama")

    class ResponseError(Exception):
        pass

    ollama.ResponseError = ResponseError
    ollama.Client = None  # tests patch this via patch.object(agent.ollama, "Client", ...)
    sys.modules["ollama"] = ollama

try:
    from dotenv import load_dotenv  # noqa: F401
except ImportError:
    # Same idea: python-dotenv isn't installed here; tests set env directly.
    import types

    dotenv = types.ModuleType("dotenv")
    dotenv.load_dotenv = lambda *a, **k: False
    sys.modules["dotenv"] = dotenv

import agent as agent_mod
import tools as tools_mod
from memory import SessionStore
from tools import Toolbox


# --- fakes -----------------------------------------------------------------
class FakeFunc:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class FakeCall:
    def __init__(self, name, arguments):
        self.function = FakeFunc(name, arguments)


class FakeMsg:
    def __init__(self, content="", tool_calls=None, thinking=None):
        self.content = content
        self.tool_calls = tool_calls or []
        self.thinking = thinking


class FakeChunk:
    def __init__(self, message):
        self.message = message


class FakeClient:
    """Scripted chat responses. Each entry: ('chunks', [...]) | ('raise', exc) | ('fact', {...})."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    def chat(self, **kw):
        self.calls += 1
        kind, payload = self.script.pop(0)
        if kind == "raise":
            raise payload
        if kind == "fact":
            class R:
                message = FakeMsg(content=json.dumps(payload))
            return R()
        return [FakeChunk(m) for m in payload]


def make_viki(script, tmp):
    store = SessionStore(folder=tmp / "session")
    with patch.object(agent_mod.ollama, "Client", return_value=FakeClient(script)):
        v = agent_mod.VIKI(Toolbox(store))
    # keep the fake client reachable for assertions
    v.client = v.client
    return v, store


def run_loop(viki, prompt, approve_ok=True):
    events = []
    out = viki.run(
        prompt, [], [], lambda k, p: events.append((k, p)),
        lambda cid, n, a: approve_ok, think=False, cancel=threading.Event(),
    )
    return out, events


class TestAgentLoop(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_tool_call_then_answer(self):
        viki, _ = make_viki([
            ("chunks", [FakeMsg(content="Let me compute. ",
                                tool_calls=[FakeCall("calculate", {"expression": "18.5*12"})])]),
            ("chunks", [FakeMsg(content="The total is 222.")]),
        ], self.tmp)
        out, events = run_loop(viki, "what is 18.5*12")
        self.assertEqual(out["type"], "text")
        self.assertIn("222", out["content"])
        self.assertTrue(any(e[0] == "tool_start" and e[1]["name"] == "calculate" for e in events))
        self.assertTrue(any(e[0] == "tool_result" and "222" in str(e[1]["result"]) for e in events))

    def test_risky_tool_needs_approval(self):
        viki, _ = make_viki([
            ("chunks", [FakeMsg(tool_calls=[FakeCall("write_file", {"path": "x.txt", "content": "hi"})])]),
            ("chunks", [FakeMsg(content="done")]),
        ], self.tmp)
        out, events = run_loop(viki, "write x", approve_ok=False)
        denied = [e for e in events if e[0] == "tool_result" and "denied" in str(e[1]["result"]).lower()]
        self.assertTrue(denied, "denied approval should reach the model as a result")

    def test_graceful_no_tool_support(self):
        viki, _ = make_viki([
            ("raise", ollama.ResponseError("model 'x' does not support tools")),
            ("chunks", [FakeMsg(content="plain answer, no tools")]),
        ], self.tmp)
        out, events = run_loop(viki, "hi")
        self.assertEqual(out["type"], "text")
        self.assertIn("plain answer", out["content"])
        self.assertTrue(any(e[0] == "status" and "no tool support" in str(e[1]) for e in events))

    def test_step_limit_wrap_up(self):
        os.environ["VIKI_MAX_STEPS"] = "2"
        try:
            viki, _ = make_viki([
                ("chunks", [FakeMsg(tool_calls=[FakeCall("current_time", {})])]),
                ("chunks", [FakeMsg(tool_calls=[FakeCall("current_time", {})])]),
                ("chunks", [FakeMsg(content="wrapped up")]),
            ], self.tmp)
            out, _ = run_loop(viki, "loop forever")
            self.assertIn("wrapped up", out["content"])
        finally:
            del os.environ["VIKI_MAX_STEPS"]

    def test_plan_event_null_steps_guarded(self):
        # Regression (2026-10-21 crash): the model sometimes returns
        # "steps": null on an update_plan call — the emitted plan payload
        # must be [], never None (frontend maps over it).
        viki, _ = make_viki([
            ("chunks", [FakeMsg(tool_calls=[FakeCall("update_plan", {"steps": None})])]),
            ("chunks", [FakeMsg(content="done")]),
        ], self.tmp)
        out, events = run_loop(viki, "plan this")
        plans = [p for k, p in events if k == "plan"]
        self.assertTrue(plans, "update_plan should emit a plan event")
        self.assertEqual(plans[0], [])

    def test_maps_tool_emits_map_event(self):
        # v0.2.27: a maps tool call must emit a "map" SSE event carrying
        # the route payload (geometry, endpoints, links) for the chat card.
        import connectors.maps as maps_mod

        def _nom(lat, lon, name):
            return [{"lat": str(lat), "lon": str(lon), "display_name": name}]

        _osrm = {"routes": [{"distance": 16093.44, "duration": 1500, "legs": [{
            "steps": [{"maneuver": {"type": "depart"}, "name": "Main St", "distance": 500},
                      {"maneuver": {"type": "arrive"}, "name": "", "distance": 0}]}],
            "geometry": {"coordinates": [[-81.9, 40.4], [-81.8, 40.45], [-81.7, 40.5]]}}]}
        _overpass = {"elements": []}
        viki, _ = make_viki([
            ("chunks", [FakeMsg(tool_calls=[FakeCall("maps_directions", {"origin": "A", "destination": "B"})])]),
            ("chunks", [FakeMsg(content="here is your route")]),
        ], self.tmp)
        with patch.object(maps_mod, "_http_get_json") as mock_get, patch.object(
            maps_mod, "_http_post", return_value=_overpass
        ), patch.dict(os.environ, {"VIKI_TOMTOM_KEY": ""}):
            mock_get.side_effect = [
                _nom(40.0, -81.9, "Start Town"),
                _nom(40.5, -81.7, "End Town"),
                _osrm,
            ]
            out, events = run_loop(viki, "directions from A to B")
        self.assertEqual(out["type"], "text")
        map_events = [p for k, p in events if k == "map"]
        self.assertTrue(map_events, "maps tool should emit a map event")
        payload = map_events[0]
        self.assertEqual(payload["geometry"][0], [40.4, -81.9])
        self.assertEqual(payload["origin"]["name"], "Start Town")
        self.assertIn("google.com/maps/dir", payload["links"]["google"])
        self.assertIsNone(payload["traffic"])


class TestTools(unittest.TestCase):
    def test_calculate(self):
        self.assertEqual(tools_mod.calculate("18.5*12 + sqrt(49)"), "229.0")
        self.assertTrue(tools_mod.calculate("__import__('os')").startswith("Cannot calculate"))

    def test_path_traversal_blocked(self):
        self.assertIn("outside the workspace", tools_mod.read_file("../../etc/passwd"))
        self.assertIn("outside the workspace", tools_mod.write_file("../evil.txt", "x"))

    def test_toolbox_unknown(self):
        tb = Toolbox(SessionStore(folder=Path(tempfile.mkdtemp()) / "s"))
        self.assertIn("Unknown tool", tb.run("nope", {}))

    def test_needs_approval(self):
        tb = Toolbox(SessionStore(folder=Path(tempfile.mkdtemp()) / "s"))
        self.assertTrue(tb.needs_approval("write_file", {"path": "a", "content": "b"}))
        self.assertFalse(tb.needs_approval("calculate", {"expression": "1+1"}))


class TestMemory(unittest.TestCase):
    def setUp(self):
        self.store = SessionStore(folder=Path(tempfile.mkdtemp()) / "s")

    def test_add_dedup(self):
        self.assertEqual(self.store.add_facts(["User likes coffee."]), ["User likes coffee."])
        self.assertEqual(self.store.add_facts(["User likes coffee."]), [])

    def test_secret_filtered(self):
        self.assertEqual(self.store.add_facts(["My api_key is abc123"]), [])
        self.assertEqual(self.store.add_facts(["SSN 123-45-6789"]), [])

    def test_remove_and_texts(self):
        self.store.add_facts(["Fact one.", "Fact two."])
        self.assertEqual(self.store.remove(1), "Fact one.")
        self.assertEqual(self.store.memory_texts(), ["Fact two."])
        self.assertIsNone(self.store.remove(99))

    def test_search_ranks_relevant_first(self):
        self.store.add_facts(["User likes black coffee.", "User drives a Honda Civic.", "User prefers Fahrenheit."])
        hits = self.store.search("should I use Fahrenheit or Celsius for the weather")
        self.assertTrue(hits)
        self.assertIn("Fahrenheit", hits[0])

    def test_search_pads_with_newest(self):
        self.store.add_facts(["User likes black coffee.", "User drives a Honda Civic."])
        hits = self.store.search("completely unrelated xyzzy query")
        # no overlap: falls back to newest facts as general context
        self.assertEqual(hits, ["User drives a Honda Civic.", "User likes black coffee."])

    def test_correction_replaces_fact(self):
        self.store.add_facts(["User lives in Millersburg Ohio."])
        added = self.store.add_facts(["User lives in Columbus Ohio now."])
        self.assertEqual(added, ["User lives in Columbus Ohio now."])
        self.assertEqual(len(self.store.memory_texts()), 1)
        self.assertIn("Columbus", self.store.memory_texts()[0])

    def test_extract_facts(self):
        viki, _ = make_viki([("fact", {"facts": ["User prefers Fahrenheit."]})], Path(tempfile.mkdtemp()))
        facts = viki.extract_facts(["I prefer Fahrenheit please"])
        self.assertEqual(facts, ["User prefers Fahrenheit."])

    def test_personality_loaded(self):
        self.assertTrue(agent_mod.PERSONALITY)
        self.assertIn("boss", agent_mod.PERSONALITY.lower())

    def test_run_accepts_callable_memory(self):
        viki, store = make_viki([("chunks", [FakeMsg(content="ok")])], Path(tempfile.mkdtemp()))
        store.add_facts(["User likes black coffee."])
        events = []
        out = viki.run("coffee order?", [], store.search, lambda k, p: events.append((k, p)),
                       lambda cid, n, a: True, think=False, cancel=threading.Event())
        self.assertEqual(out["type"], "text")


if __name__ == "__main__":
    unittest.main(verbosity=2)
