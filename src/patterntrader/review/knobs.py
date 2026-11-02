"""The settings the reviewer may adjust, the limits it can't leave, and loading its adjustments.

Only settings listed under `reviewer.knobs` in config.yaml can change, each within its min/max and
by at most one step per change. Everything else (paper-only keys, GitHub-only orders, the 15:50
flatten, shorts, sentiment, the edge gate) is out of reach. Adjustments live in
reviewer/overrides.yaml on the trading-log branch and are re-checked against the limits every
time they load, so an edited or damaged file can't loosen the rules.
"""

from __future__ import annotations

import copy
import logging
import re
from pathlib import Path
from typing import Any

import yaml

from patterntrader.config import ROOT, Cfg

log = logging.getLogger("review")
MISSING = object()


def state_dir(cfg, root: Path | None = None) -> Path:
    return (root or ROOT) / cfg.reviewer.get("state_dir", "reviewer")


def get_path(cfg, path: str) -> Any:
    node = cfg
    for k in path.split("."):
        if not isinstance(node, dict) or k not in node:
            return MISSING
        node = node[k]
    return node


def _set_path(d: dict, path: str, value: Any) -> None:
    *head, last = path.split(".")
    for k in head:
        d = d[k]
    d[last] = value


def spec_for(cfg, knob: str) -> dict | None:
    """The limits for `knob`, or None if the reviewer may not touch it."""
    rc = cfg.reviewer
    parts = knob.split(".")
    if parts[0] == "strategies":
        if len(parts) != 3 or parts[1] not in rc.strategies:
            return None
        spec = rc.knobs.get(f"strategies.*.{parts[2]}")
    else:
        spec = rc.knobs.get(knob)
    if spec is None or get_path(cfg, knob) is MISSING:
        return None
    return spec


def all_knobs(cfg) -> list[str]:
    out = []
    for pattern in cfg.reviewer.knobs:
        if pattern.startswith("strategies.*."):
            out += [f"strategies.{s}.{pattern.split('.', 2)[2]}" for s in cfg.reviewer.strategies]
        else:
            out.append(pattern)
    return [k for k in out if spec_for(cfg, k) is not None]


def _minutes(t: str) -> int:
    hh, mm = t.split(":")
    return int(hh) * 60 + int(mm)


def check_value(spec: dict, current: Any, value: Any, step: bool = True) -> tuple[bool, str, Any]:
    """Is `value` inside the knob's limits? With step=True it may also move only one step from `current`."""
    if spec.get("bool"):
        if not isinstance(value, bool):
            return False, "must be true or false", None
        return True, "", value

    if "subset_of" in spec:
        allowed = list(spec["subset_of"])
        if not isinstance(value, list) or not value or any(v not in allowed for v in value):
            return False, f"must be a non-empty list drawn from {allowed}", None
        value = [v for v in allowed if v in value]
        if step and len(set(value) ^ set(current)) > 1:
            return False, "add or remove one market type at a time", None
        return True, "", value

    if "window" in spec:
        lo, hi = map(_minutes, spec["window"])
        if (not isinstance(value, list) or len(value) != 2
                or not all(isinstance(v, str) and re.fullmatch(r"\d{2}:\d{2}", v) for v in value)):
            return False, 'must be ["HH:MM", "HH:MM"]', None
        a, b = map(_minutes, value)
        if not lo <= a < b <= hi:
            return False, f"must sit inside {spec['window'][0]}-{spec['window'][1]} with start before end", None
        if step and max(abs(a - _minutes(current[0])), abs(b - _minutes(current[1]))) > spec["step_minutes"]:
            return False, f"each end may move at most {spec['step_minutes']} minutes per change", None
        return True, "", list(value)

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False, "must be a number", None
    if isinstance(current, int) and not isinstance(current, bool):
        if value != int(value):
            return False, "must be a whole number", None
        value = int(value)
    if not spec["min"] <= value <= spec["max"]:
        return False, f"must be between {spec['min']} and {spec['max']}", None
    if step and abs(value - current) > spec["step"] + 1e-9:
        return False, f"may move at most {spec['step']} per change (now {current})", None
    return True, "", value


def load_overrides(cfg, root: Path | None = None) -> dict:
    p = state_dir(cfg, root) / "overrides.yaml"
    return (yaml.safe_load(p.read_text()) or {}) if p.exists() else {}


def save_overrides(cfg, overrides: dict, root: Path | None = None) -> None:
    p = state_dir(cfg, root) / "overrides.yaml"
    p.parent.mkdir(parents=True, exist_ok=True)
    head = ("# The reviewer's adjustments to config.yaml, applied to every scan and trading day.\n"
            "# Delete a line (or the file) to go back to config.yaml's value.\n")
    p.write_text(head + yaml.safe_dump(overrides, sort_keys=True, default_flow_style=None))


def with_overrides(cfg, overrides: dict) -> Cfg:
    """config.yaml plus these adjustments; any outside the limits are skipped and logged."""
    out = copy.deepcopy(cfg)
    for knob, value in overrides.items():
        spec = spec_for(cfg, knob)
        ok, why, v = (False, "not a setting the reviewer may change", None) if spec is None \
            else check_value(spec, get_path(cfg, knob), value, step=False)
        if ok:
            _set_path(out, knob, v)
        else:
            log.warning("ignoring reviewer adjustment %s=%r: %s", knob, value, why)
    return Cfg.wrap(out)


def apply_overrides(cfg, root: Path | None = None) -> Cfg:
    if not cfg.get("reviewer", {}).get("enabled"):
        return cfg
    over = load_overrides(cfg, root)
    if over:
        log.info("reviewer adjustments in effect: %s", ", ".join(f"{k}={v}" for k, v in over.items()))
    return with_overrides(cfg, over) if over else cfg
