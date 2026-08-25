"""Build monitors without importing controller dependencies early."""

from typing import Any


def build_monitor(*args: Any, **kwargs: Any):
    """Build one configured monitor."""
    from avalanche.monitors.factory import build_monitor as factory

    return factory(*args, **kwargs)


def __getattr__(name: str):
    """Load monitor classes only when a caller requests them."""
    if name in {"AllowMonitor", "OutcomeMonitor"}:
        from avalanche.monitors import outcome

        return getattr(outcome, name)
    if name in {"RuleMonitor", "RulePrediction"}:
        from avalanche.monitors import rules

        return getattr(rules, name)
    if name in {"FEATURE_NAMES", "FEATURE_VERSION", "FeatureExtractor"}:
        from avalanche.monitors import features

        return getattr(features, name)
    raise AttributeError(name)


__all__ = [
    "FEATURE_NAMES",
    "FEATURE_VERSION",
    "AllowMonitor",
    "FeatureExtractor",
    "OutcomeMonitor",
    "RuleMonitor",
    "RulePrediction",
    "build_monitor",
]
