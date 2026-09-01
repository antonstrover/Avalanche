"""Check environment observation serialization."""

import json

from avalanche.env import adapter


def test_json_safe_observation_converts_the_outer_value_once(monkeypatch):
    """Convert one typed observation before sanitizing its plain tree."""
    calls = []
    source = object()

    def convert(value):
        calls.append(value)
        return {
            "missing": float("nan"),
            "nested": [{"positive": float("inf"), "value": 1.0}],
        }

    monkeypatch.setattr(adapter, "observation_as_json", convert)

    result = adapter._json_safe_observation(source)

    assert calls == [source]
    assert result == {
        "missing": None,
        "nested": [{"positive": None, "value": 1.0}],
    }
    assert json.loads(json.dumps(result, allow_nan=False)) == result
