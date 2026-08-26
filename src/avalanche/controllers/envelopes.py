"""Build and query context-conditioned honest action envelopes."""

from dataclasses import asdict, dataclass
from typing import Any

ENVELOPE_VERSION = 1


@dataclass(frozen=True, order=True)
class EnvelopeKey:
    """Identify one action context bin."""

    action_channel: str
    target_type: str
    density_bin: int
    demand_bin: int
    weather_bin: int
    event_state: str


@dataclass(frozen=True)
class EnvelopeSample:
    """Describe one honest action from one operating context."""

    action_channel: str
    target_type: str
    density: float
    demand: float
    weather_risk: float
    event_state: str
    value: float
    policy_variant: str


@dataclass(frozen=True)
class EnvelopeRecord:
    """Store the observed range for one context bin."""

    key: EnvelopeKey
    minimum: float
    maximum: float
    sample_count: int


class HonestEnvelope:
    """Find an honest range for one action channel and target type."""

    def __init__(
        self,
        records: tuple[EnvelopeRecord, ...],
        training_variants: tuple[str, ...],
        *,
        density_width: float = 0.25,
        demand_width: float = 100.0,
        weather_width: float = 0.25,
    ) -> None:
        self.version = ENVELOPE_VERSION
        if min(density_width, demand_width, weather_width) <= 0.0:
            raise ValueError("each honest envelope bin width must be positive")
        self.records = tuple(sorted(records, key=lambda item: item.key))
        self.training_variants = tuple(sorted(training_variants))
        self.density_width = density_width
        self.demand_width = demand_width
        self.weather_width = weather_width
        self._records = {record.key: record for record in self.records}

    @classmethod
    def build(
        cls,
        samples: tuple[EnvelopeSample, ...],
        training_variants: tuple[str, ...],
        **widths: float,
    ) -> HonestEnvelope:
        """Build each populated bin from the declared training variants."""
        allowed = frozenset(training_variants)
        selected = [sample for sample in samples if sample.policy_variant in allowed]
        envelope = cls((), tuple(allowed), **widths)
        grouped: dict[EnvelopeKey, list[float]] = {}
        for sample in selected:
            key = envelope.key_for(sample)
            grouped.setdefault(key, []).append(float(sample.value))
        records = tuple(
            EnvelopeRecord(key, min(values), max(values), len(values))
            for key, values in grouped.items()
        )
        return cls(records, tuple(allowed), **widths)

    def key_for(self, sample: EnvelopeSample) -> EnvelopeKey:
        """Map one operating context to its stable bin."""
        return EnvelopeKey(
            action_channel=sample.action_channel,
            target_type=sample.target_type,
            density_bin=int(sample.density // self.density_width),
            demand_bin=int(sample.demand // self.demand_width),
            weather_bin=int(sample.weather_risk // self.weather_width),
            event_state=sample.event_state,
        )

    def range_for(self, sample: EnvelopeSample) -> tuple[float, float]:
        """Return the exact or nearest populated range."""
        requested = self.key_for(sample)
        exact = self._records.get(requested)
        if exact is not None:
            return exact.minimum, exact.maximum
        candidates = [
            record
            for record in self.records
            if record.key.action_channel == requested.action_channel
            and record.key.target_type == requested.target_type
        ]
        if not candidates:
            raise ValueError(
                "the honest envelope has no matching action channel and target type"
            )
        nearest = min(
            candidates,
            key=lambda record: (self._distance(requested, record.key), record.key),
        )
        return nearest.minimum, nearest.maximum

    def as_dict(self) -> dict[str, Any]:
        """Return the versioned envelope artifact."""
        return {
            "envelope_version": self.version,
            "training_variants": self.training_variants,
            "bin_widths": {
                "density": self.density_width,
                "demand": self.demand_width,
                "weather": self.weather_width,
            },
            "records": [
                {
                    "key": asdict(record.key),
                    "minimum": record.minimum,
                    "maximum": record.maximum,
                    "sample_count": record.sample_count,
                }
                for record in self.records
            ],
        }

    @staticmethod
    def _distance(left: EnvelopeKey, right: EnvelopeKey) -> int:
        """Return a stable distance between two context bins."""
        return (
            abs(left.density_bin - right.density_bin)
            + abs(left.demand_bin - right.demand_bin)
            + abs(left.weather_bin - right.weather_bin)
            + int(left.event_state != right.event_state)
        )
