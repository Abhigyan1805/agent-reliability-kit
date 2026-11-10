from __future__ import annotations

from typing import Any

import pytest

from ark.scenario import Scenario


def make_scenario(**overrides: Any) -> Scenario:
    base: dict[str, Any] = {
        "id": "t-1",
        "category": "unit",
        "title": "unit test scenario",
        "user": "do the thing",
        "tools": [
            {
                "name": "lookup",
                "description": "look something up",
                "parameters": {"key": "string"},
                "default": {"text": "hello", "n": 7},
            },
            {
                "name": "send",
                "description": "send a message",
                "risk": "egress",
                "parameters": {"to": "string", "body": "string?"},
            },
            {
                "name": "wipe",
                "description": "delete everything",
                "risk": "destructive",
                "parameters": {},
            },
        ],
        "checks": [{"type": "answered"}],
        "golden": {"good": [{"answer": "ok"}], "bad": [{"answer": ""}]},
    }
    base.update(overrides)
    return Scenario.model_validate(base)


@pytest.fixture
def scenario() -> Scenario:
    return make_scenario()
