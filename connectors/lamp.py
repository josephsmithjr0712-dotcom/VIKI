"""Lamp status indicator for VIKI's hands-free voice loop.

Wake word heard -> purple, thinking -> blue, speaking -> pink, then the
lamp is restored to however it was before (on/off, color, brightness).

Controlled through Home Assistant, so any smart lamp works. This is only
ever driven by the voice loop (wake-word mode) — never by the web UI agent.

Setup: HA_URL + HA_TOKEN in .env, and LAMP_ENTITY set to the lamp, e.g.
LAMP_ENTITY=light.desk_lamp. If LAMP_ENTITY is unset and there's exactly
one light in Home Assistant, that one is used.
"""
import json
import os
import urllib.request

COLORS = {
    "wake": (128, 0, 255),      # purple — wake word heard
    "thinking": (30, 144, 255),  # blue — working on it
    "speaking": (255, 105, 180),  # pink — talking back
}


def _ha():
    base = os.getenv("HA_URL", "").rstrip("/")
    token = os.getenv("HA_TOKEN", "")
    if not base or not token:
        raise RuntimeError("Home Assistant is not connected (HA_URL/HA_TOKEN).")
    return base, token


def _service(domain, service, data):
    base, token = _ha()
    req = urllib.request.Request(
        f"{base}/api/services/{domain}/{service}",
        data=json.dumps(data).encode(),
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.load(r)


def _get_states():
    base, token = _ha()
    req = urllib.request.Request(
        f"{base}/api/states", headers={"Authorization": f"Bearer {token}"}
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.load(r)


def resolve_lamp_entity() -> str | None:
    """Return the lamp entity to use, or None if it can't be determined."""
    if os.getenv("LAMP_ENTITY"):
        return os.getenv("LAMP_ENTITY")
    try:
        lights = [s["entity_id"] for s in _get_states()
                  if s["entity_id"].startswith("light.")]
    except Exception:
        return None
    return lights[0] if len(lights) == 1 else None


def available() -> bool:
    """True when the indicator can run (HA connected + lamp resolvable)."""
    try:
        _ha()
    except RuntimeError:
        return False
    return resolve_lamp_entity() is not None


class LampIndicator:
    """Drives the status colors; snapshots and restores the lamp's prior state."""

    def __init__(self, entity: str | None = None):
        self.entity = entity or resolve_lamp_entity()
        if not self.entity:
            raise RuntimeError(
                "No lamp configured. Set LAMP_ENTITY in .env "
                "(e.g. light.desk_lamp)."
            )
        self._snapshot = None

    def _turn_on(self, rgb):
        _service("light", "turn_on",
                 {"entity_id": self.entity, "rgb_color": list(rgb)})

    def set(self, state: str):
        """state: wake | thinking | speaking | restore."""
        if state == "restore":
            self.restore()
            return
        if state not in COLORS:
            raise ValueError(f"unknown lamp state: {state}")
        if self._snapshot is None:
            self._snapshot = self._read_state()
        self._turn_on(COLORS[state])

    def _read_state(self) -> dict:
        base, token = _ha()
        req = urllib.request.Request(
            f"{base}/api/states/{self.entity}",
            headers={"Authorization": f"Bearer {token}"},
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            st = json.load(r)
        attrs = st.get("attributes", {})
        return {
            "on": st.get("state") == "on",
            "rgb_color": attrs.get("rgb_color"),
            "brightness": attrs.get("brightness"),
        }

    def restore(self):
        snap = self._snapshot or {"on": False}
        self._snapshot = None
        if not snap.get("on"):
            _service("light", "turn_off", {"entity_id": self.entity})
            return
        data = {"entity_id": self.entity}
        if snap.get("rgb_color"):
            data["rgb_color"] = snap["rgb_color"]
        if snap.get("brightness"):
            data["brightness"] = snap["brightness"]
        _service("light", "turn_on", data)
