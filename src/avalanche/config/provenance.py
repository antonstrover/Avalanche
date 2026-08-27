"""Describe the source of each resolved configuration value."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ValueProvenance(BaseModel):
    """Record one explicit, defaulted, or derived value source."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    pointer: str
    kind: Literal["explicit", "schema_default", "derived"]
    owner: Literal[
        "mountain", "scenario", "controller", "monitor", "override", "resolver"
    ]
    source_path: str | None = None
    line: int | None = Field(default=None, ge=1)
    column: int | None = Field(default=None, ge=1)
    source_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    schema_path: str | None = None
    formula_version: str | None = None
    input_paths: tuple[str, ...] = ()
