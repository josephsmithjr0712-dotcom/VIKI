"""Tests for the home_assistant tool. HA's REST API is mocked (no network)."""
import copy
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tools as tools_mod

STATES = [
    {"entity_id": "light.living_room", "state": "on",
     "attributes": {"friendly_name": "Living Room", "brightness": 200,
                    "supported_color_modes": ["brightness"]}},
    {"entity_id": "switch.fan", "state": "off",
     "attributes": {"friendly_name": "Fan"}},
    {"entity_id": "media_player.tv", "state": "idle",
     "attributes": {"friendly_name": "TV", "volume_level": 0.5}},
]


def fresh_cache(full=None):
    """Seed HA_ENTITIES_CACHE the way refresh_ha_entities() would."""
    ents = [{"entity_id": s["entity_id"],
             "domain": s["entity_id"].split(".")[0],
             "name": s.get("attributes", {}).get("friendly_name", s["entity_id"]),
             "state": s["state"]} for s in STATES]
    return {"updated": "2026-10-23_09:00:00", "entities": ents,
            "full": copy.deepcopy(STATES) if full is None else full}


class HomeAssistantTests(unittest.TestCase):
    def setUp(self):
        self._orig = copy.deepcopy(tools_mod.HA_ENTITIES_CACHE)
        tools_mod.HA_ENTITIES_CACHE.clear()
        tools_mod.HA_ENTITIES_CACHE.update(fresh_cache())

    def tearDown(self):
        tools_mod.HA_ENTITIES_CACHE.clear()
        tools_mod.HA_ENTITIES_CACHE.update(self._orig)

    def test_list_uses_cache_without_network(self):
        with patch.object(tools_mod, "_ha_request") as req:
            out = tools_mod.home_assistant("list")
        req.assert_not_called()
        for eid in ("light.living_room", "switch.fan", "media_player.tv"):
            self.assertIn(eid, out)

    def test_list_refreshes_empty_cache(self):
        tools_mod.HA_ENTITIES_CACHE.clear()
        tools_mod.HA_ENTITIES_CACHE.update({"updated": None, "entities": [], "full": []})
        with patch.object(tools_mod, "_ha_request", return_value=STATES) as req:
            out = tools_mod.home_assistant("list")
        req.assert_called_once_with("/api/states", timeout=15)
        self.assertIn("light.living_room", out)
        self.assertTrue(tools_mod.HA_ENTITIES_CACHE["entities"])

    def test_list_unreachable(self):
        tools_mod.HA_ENTITIES_CACHE.clear()
        tools_mod.HA_ENTITIES_CACHE.update({"updated": None, "entities": [], "full": []})
        with patch.object(tools_mod, "_ha_request", side_effect=Exception("no route")):
            out = tools_mod.home_assistant("list")
        self.assertIn("Could not reach", out)

    def test_dump_one_entity(self):
        out = tools_mod.home_assistant("dump", entity_id="switch.fan")
        data = json.loads(out)
        self.assertEqual(data["entity_id"], "switch.fan")
        self.assertEqual(data["state"], "off")

    def test_dump_entity_not_found(self):
        out = tools_mod.home_assistant("dump", entity_id="light.nope")
        self.assertIn("not found", out)

    def test_dump_all_trimmed(self):
        big = [{"entity_id": f"light.n{i}", "state": "off",
                "attributes": {"note": "x" * 2000}} for i in range(50)]
        tools_mod.HA_ENTITIES_CACHE.clear()
        tools_mod.HA_ENTITIES_CACHE.update(fresh_cache(full=big))
        out = tools_mod.home_assistant("dump")
        self.assertLessEqual(len(out), 12000)
        self.assertIn("light.n0", out)

    def test_state(self):
        with patch.object(tools_mod, "_ha_request",
                          return_value=STATES[0]) as req:
            out = tools_mod.home_assistant("state", entity_id="light.living_room")
        req.assert_called_once_with("/api/states/light.living_room")
        data = json.loads(out)
        self.assertEqual(data["state"], "on")
        self.assertEqual(data["attributes"]["brightness"], 200)

    def test_call(self):
        with patch.object(tools_mod, "_ha_request", return_value=[]) as req:
            out = tools_mod.home_assistant("call", entity_id="light.living_room",
                                           domain="light", service="turn_off")
        req.assert_called_once_with("/api/services/light/turn_off",
                                    {"entity_id": "light.living_room"})
        self.assertEqual(json.loads(out), [])

    def test_unknown_action_lists_all_four(self):
        out = tools_mod.home_assistant("bogus")
        for a in ("list", "state", "dump", "call"):
            self.assertIn(a, out)

    def test_request_exception_wrapped(self):
        with patch.object(tools_mod, "_ha_request",
                          side_effect=ConnectionError("refused")):
            out = tools_mod.home_assistant("state", entity_id="light.living_room")
        self.assertIn("Home Assistant error", out)
        self.assertIn("refused", out)

    def test_entities_text_empty_cache(self):
        tools_mod.HA_ENTITIES_CACHE.clear()
        tools_mod.HA_ENTITIES_CACHE.update({"updated": None, "entities": [], "full": []})
        self.assertEqual(tools_mod.ha_entities_text(), "")

    def test_entities_text_lists_all(self):
        text = tools_mod.ha_entities_text()
        self.assertIn("light.living_room (Living Room): on", text)


class HomeAssistantApprovalGateTests(unittest.TestCase):
    """lock/alarm_control_panel/cover `call` actions need approval again;
    everything else in home_assistant acts directly."""

    def setUp(self):
        self.tb = tools_mod.Toolbox(store=None)

    def call(self, domain, service="turn_on", entity_id="x.y", action="call"):
        return self.tb.needs_approval(
            "home_assistant",
            {"action": action, "domain": domain, "service": service,
             "entity_id": entity_id})

    def test_lock_unlock_needs_approval(self):
        self.assertTrue(self.call("lock", "unlock", "lock.front_door"))

    def test_lock_lock_needs_approval(self):
        self.assertTrue(self.call("lock", "lock", "lock.front_door"))

    def test_alarm_arm_needs_approval(self):
        self.assertTrue(
            self.call("alarm_control_panel", "alarm_arm_away", "alarm_control_panel.home"))

    def test_alarm_disarm_needs_approval(self):
        self.assertTrue(
            self.call("alarm_control_panel", "alarm_disarm", "alarm_control_panel.home"))

    def test_cover_open_needs_approval(self):
        self.assertTrue(self.call("cover", "open_cover", "cover.garage_door"))

    def test_domain_case_insensitive(self):
        self.assertTrue(self.call("Lock", "unlock", "lock.front_door"))

    def test_light_call_no_approval(self):
        self.assertFalse(self.call("light", "turn_on", "light.living_room"))

    def test_switch_call_no_approval(self):
        self.assertFalse(self.call("switch", "toggle", "switch.fan"))

    def test_climate_call_no_approval(self):
        self.assertFalse(self.call("climate", "set_temperature", "climate.thermostat"))

    def test_list_state_dump_no_approval(self):
        for action in ("list", "state", "dump"):
            self.assertFalse(
                self.tb.needs_approval(
                    "home_assistant",
                    {"action": action, "entity_id": "lock.front_door"}),
                f"action={action} should not need approval")

    def test_call_without_domain_no_approval(self):
        self.assertFalse(
            self.tb.needs_approval("home_assistant", {"action": "call"}))

    def test_other_tools_unaffected(self):
        self.assertTrue(
            self.tb.needs_approval("write_file", {"path": "a.txt", "content": "x"}))
        self.assertFalse(
            self.tb.needs_approval("web_search", {"query": "weather"}))
        self.assertFalse(
            self.tb.needs_approval("no_such_tool", {}))


if __name__ == "__main__":
    unittest.main()
