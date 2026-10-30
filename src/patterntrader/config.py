"""Config loading and the paper-only credential guard."""

from __future__ import annotations

import os
from datetime import time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import yaml

ET = ZoneInfo("America/New_York")
ROOT = Path(__file__).resolve().parents[2]


class Cfg(dict):
    """Dict with attribute access, recursively applied to nested dicts."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    @classmethod
    def wrap(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return cls({k: cls.wrap(v) for k, v in value.items()})
        if isinstance(value, list):
            return [cls.wrap(v) for v in value]
        return value


def load_config(path: str | Path | None = None) -> Cfg:
    path = Path(path) if path else ROOT / "config.yaml"
    with open(path) as fh:
        return Cfg.wrap(yaml.safe_load(fh))


def parse_time(value: str) -> time:
    hh, mm = value.split(":")
    return time(int(hh), int(mm))


class LiveKeyError(RuntimeError):
    pass


def paper_credentials() -> tuple[str, str]:
    """Return Alpaca keys, refusing anything that isn't a paper key."""
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    key = os.environ.get("ALPACA_API_KEY", "").strip()
    secret = os.environ.get("ALPACA_SECRET_KEY", "").strip()
    if not key or not secret:
        raise LiveKeyError("ALPACA_API_KEY / ALPACA_SECRET_KEY are not set (see .env.example).")
    assert_paper_key(key)
    return key, secret


def assert_paper_key(key: str) -> None:
    # Alpaca paper keys start with "PK"; live keys start with "AK".
    if not key.startswith("PK"):
        raise LiveKeyError(
            "Refusing to run: this does not look like an Alpaca PAPER key (paper keys start "
            "with 'PK'). This bot is paper-trading only."
        )
