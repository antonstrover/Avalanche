"""Encode canonical values and calculate formal SHA-256 identities."""

from __future__ import annotations

import hashlib
import math
import struct
from collections.abc import Mapping, Sequence
from typing import Any

import msgpack
import numpy as np

CHECKSUM_FIELD_NAMES = frozenset(
    {
        "artifact_sha256",
        "continuation_checksum",
        "physical_state_checksum",
        "state_checksum",
    }
)

_TYPE = "$type"
_CANONICAL_NAN64 = 0x7FF8000000000000
_CANONICAL_NAN32 = 0x7FC00000


class CanonicalEncodingError(ValueError):
    """Report a value that the canonical encoder cannot represent."""


def canonical_messagepack(value: Any, *, allow_nonfinite: bool = False) -> bytes:
    """Return deterministic MessagePack bytes for one supported value."""
    encoded = _encode(value, allow_nonfinite=allow_nonfinite)
    return msgpack.packb(encoded, use_bin_type=True, strict_types=True)


def decode_canonical_messagepack(
    content: bytes, *, allow_nonfinite: bool = False
) -> Any:
    """Decode canonical MessagePack and reject a noncanonical representation."""
    try:
        encoded = msgpack.unpackb(
            content,
            raw=False,
            strict_map_key=False,
        )
    except (msgpack.UnpackException, ValueError) as error:
        raise CanonicalEncodingError("the MessagePack payload is invalid") from error
    value = _decode(encoded, allow_nonfinite=allow_nonfinite)
    if canonical_messagepack(value, allow_nonfinite=allow_nonfinite) != content:
        raise CanonicalEncodingError("the MessagePack payload is not canonical")
    return value


def canonical_sha256(value: Any, *, allow_nonfinite: bool = False) -> str:
    """Return the SHA-256 identity of one canonical value."""
    return hashlib.sha256(
        canonical_messagepack(value, allow_nonfinite=allow_nonfinite)
    ).hexdigest()


def checksum_domain(value: Any) -> Any:
    """Copy one value without any checksum field in its mappings."""
    if isinstance(value, Mapping):
        return {
            key: checksum_domain(item)
            for key, item in value.items()
            if key not in CHECKSUM_FIELD_NAMES
        }
    if isinstance(value, tuple):
        return tuple(checksum_domain(item) for item in value)
    if isinstance(value, list):
        return [checksum_domain(item) for item in value]
    return value


def named_checksum(value: Any, *, allow_nonfinite: bool = False) -> str:
    """Return a SHA-256 identity after checksum fields self-exclude."""
    return canonical_sha256(
        checksum_domain(value),
        allow_nonfinite=allow_nonfinite,
    )


def _encode(value: Any, *, allow_nonfinite: bool) -> Any:
    if value is None:
        return {_TYPE: "null"}
    if isinstance(value, (bool, np.bool_)):
        return {_TYPE: "boolean", "value": bool(value)}
    if isinstance(value, (int, np.integer)):
        return _encode_integer(int(value))
    if isinstance(value, (float, np.floating)):
        return _encode_float(float(value), allow_nonfinite=allow_nonfinite)
    if isinstance(value, str):
        return {_TYPE: "string", "data": _length_prefix(value.encode("utf-8"))}
    if isinstance(value, bytes):
        return {_TYPE: "bytes", "data": _length_prefix(value)}
    if isinstance(value, np.ndarray):
        return _encode_array(value, allow_nonfinite=allow_nonfinite)
    if isinstance(value, Mapping):
        keys = tuple(value)
        if any(not isinstance(key, str) for key in keys):
            raise CanonicalEncodingError("a canonical mapping key must be a string")
        ordered = sorted(keys, key=lambda item: item.encode("utf-8"))
        return {
            key: _encode(value[key], allow_nonfinite=allow_nonfinite)
            for key in ordered
        }
    if isinstance(value, tuple):
        return {
            _TYPE: "tuple",
            "items": [
                _encode(item, allow_nonfinite=allow_nonfinite) for item in value
            ],
        }
    if isinstance(value, list):
        return {
            _TYPE: "list",
            "items": [
                _encode(item, allow_nonfinite=allow_nonfinite) for item in value
            ],
        }
    raise CanonicalEncodingError(
        f"the canonical value type {type(value).__name__!r} is unsupported"
    )


def _encode_integer(value: int) -> dict[str, Any]:
    sign = -1 if value < 0 else int(value > 0)
    magnitude_value = abs(value)
    width = (magnitude_value.bit_length() + 7) // 8
    magnitude = magnitude_value.to_bytes(width, "big") if width else b""
    return {
        _TYPE: "integer",
        "magnitude": _length_prefix(magnitude),
        "sign": sign,
    }


def _encode_float(value: float, *, allow_nonfinite: bool) -> dict[str, Any]:
    if math.isnan(value):
        if not allow_nonfinite:
            raise CanonicalEncodingError("a formal value must not be NaN")
        return {_TYPE: "quiet-nan"}
    if math.isinf(value):
        if not allow_nonfinite:
            raise CanonicalEncodingError("a formal value must be finite")
        return {_TYPE: "positive-infinity" if value > 0.0 else "negative-infinity"}
    return {_TYPE: "float64", "data": _length_prefix(struct.pack("<d", value))}


def _encode_array(
    values: np.ndarray, *, allow_nonfinite: bool
) -> dict[str, Any]:
    source = np.asarray(values)
    if source.dtype.kind not in "biuf":
        raise CanonicalEncodingError(f"the array type {source.dtype.str!r} is unsupported")
    dtype = _little_endian_dtype(source.dtype)
    portable = np.ascontiguousarray(source, dtype=dtype)
    if portable.dtype.kind == "f":
        finite = np.isfinite(portable)
        if not allow_nonfinite and not np.all(finite):
            raise CanonicalEncodingError("a formal array must contain finite values")
        portable = _canonicalize_array_nan(portable)
    data = portable.tobytes(order="C")
    return {
        _TYPE: "ndarray",
        "data": _length_prefix(data),
        "dtype": dtype.str,
        "length": _encode_integer(portable.size),
        "shape": [_encode_integer(int(size)) for size in portable.shape],
    }


def _decode(value: Any, *, allow_nonfinite: bool) -> Any:
    if isinstance(value, list):
        return [_decode(item, allow_nonfinite=allow_nonfinite) for item in value]
    if not isinstance(value, dict):
        raise CanonicalEncodingError("a canonical value must use a tagged mapping")
    if any(not isinstance(key, str) for key in value):
        raise CanonicalEncodingError("a canonical mapping key must be a string")
    keys = tuple(value)
    if keys != tuple(sorted(keys, key=lambda item: item.encode("utf-8"))):
        raise CanonicalEncodingError("canonical mapping keys are not ordered")
    tag = value.get(_TYPE)
    if tag is None:
        return {
            key: _decode(item, allow_nonfinite=allow_nonfinite)
            for key, item in value.items()
        }
    if tag == "null" and keys == (_TYPE,):
        return None
    if tag == "boolean" and set(value) == {_TYPE, "value"}:
        if not isinstance(value["value"], bool):
            raise CanonicalEncodingError("the canonical Boolean is invalid")
        return value["value"]
    if tag == "integer":
        return _decode_integer(value)
    if tag == "float64":
        _require_fields(value, {_TYPE, "data"}, "float")
        data = _remove_length_prefix(value["data"])
        if len(data) != 8:
            raise CanonicalEncodingError("the canonical float has a bad length")
        result = struct.unpack("<d", data)[0]
        if not math.isfinite(result):
            raise CanonicalEncodingError("the canonical finite float is invalid")
        return result
    if tag == "quiet-nan" and keys == (_TYPE,):
        if not allow_nonfinite:
            raise CanonicalEncodingError("a formal value must not be NaN")
        return float("nan")
    if tag in {"positive-infinity", "negative-infinity"} and keys == (_TYPE,):
        if not allow_nonfinite:
            raise CanonicalEncodingError("a formal value must be finite")
        return float("inf") if tag == "positive-infinity" else float("-inf")
    if tag in {"string", "bytes"}:
        _require_fields(value, {_TYPE, "data"}, tag)
        data = _remove_length_prefix(value["data"])
        if tag == "bytes":
            return data
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError as error:
            raise CanonicalEncodingError("the canonical string is not UTF-8") from error
    if tag in {"tuple", "list"}:
        _require_fields(value, {_TYPE, "items"}, tag)
        items = value["items"]
        if not isinstance(items, list):
            raise CanonicalEncodingError(f"the canonical {tag} items are invalid")
        decoded = [_decode(item, allow_nonfinite=allow_nonfinite) for item in items]
        return tuple(decoded) if tag == "tuple" else decoded
    if tag == "ndarray":
        return _decode_array(value, allow_nonfinite=allow_nonfinite)
    raise CanonicalEncodingError(f"the canonical tag {tag!r} is unsupported")


def _decode_integer(value: dict[str, Any]) -> int:
    _require_fields(value, {_TYPE, "magnitude", "sign"}, "integer")
    sign = value["sign"]
    if not isinstance(sign, int) or isinstance(sign, bool) or sign not in {-1, 0, 1}:
        raise CanonicalEncodingError("the canonical integer sign is invalid")
    magnitude = _remove_length_prefix(value["magnitude"])
    if magnitude.startswith(b"\x00"):
        raise CanonicalEncodingError("the canonical integer magnitude has a leading zero")
    if sign == 0 and magnitude:
        raise CanonicalEncodingError("a zero integer must have an empty magnitude")
    if sign != 0 and not magnitude:
        raise CanonicalEncodingError("a nonzero integer needs a magnitude")
    result = int.from_bytes(magnitude, "big")
    return result * sign


def _decode_array(value: dict[str, Any], *, allow_nonfinite: bool) -> np.ndarray:
    _require_fields(
        value,
        {_TYPE, "data", "dtype", "length", "shape"},
        "array",
    )
    dtype_name = value["dtype"]
    if not isinstance(dtype_name, str):
        raise CanonicalEncodingError("the canonical array type is invalid")
    try:
        dtype = np.dtype(dtype_name)
    except TypeError as error:
        raise CanonicalEncodingError("the canonical array type is invalid") from error
    if dtype.kind not in "biuf" or dtype != _little_endian_dtype(dtype):
        raise CanonicalEncodingError("the canonical array type is unsupported")
    shape_value = value["shape"]
    if not isinstance(shape_value, list):
        raise CanonicalEncodingError("the canonical array shape is invalid")
    shape = tuple(_decode_integer(item) for item in shape_value)
    if any(size < 0 for size in shape):
        raise CanonicalEncodingError("the canonical array shape is invalid")
    length = _decode_integer(value["length"])
    if length != math.prod(shape):
        raise CanonicalEncodingError("the canonical array length is invalid")
    data = _remove_length_prefix(value["data"])
    if len(data) != length * dtype.itemsize:
        raise CanonicalEncodingError("the canonical array bytes have a bad length")
    result = np.frombuffer(data, dtype=dtype).reshape(shape).copy()
    if result.dtype.kind == "f":
        if not allow_nonfinite and not np.all(np.isfinite(result)):
            raise CanonicalEncodingError("a formal array must contain finite values")
        if not np.array_equal(
            result.view(np.uint8),
            _canonicalize_array_nan(result).view(np.uint8),
        ):
            raise CanonicalEncodingError("the canonical array has a noncanonical NaN")
    return result


def _little_endian_dtype(dtype: np.dtype[Any]) -> np.dtype[Any]:
    value = np.dtype(dtype)
    if value.itemsize == 1:
        return value.newbyteorder("|")
    return value.newbyteorder("<")


def _canonicalize_array_nan(values: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(values).copy()
    nan = np.isnan(result)
    if not np.any(nan):
        return result
    if result.dtype.itemsize == 8:
        result.view(np.uint64)[nan] = _CANONICAL_NAN64
    elif result.dtype.itemsize == 4:
        result.view(np.uint32)[nan] = _CANONICAL_NAN32
    else:
        result[nan] = np.nan
    return result


def _length_prefix(payload: bytes) -> bytes:
    return struct.pack("<Q", len(payload)) + payload


def _remove_length_prefix(value: Any) -> bytes:
    if not isinstance(value, bytes) or len(value) < 8:
        raise CanonicalEncodingError("a tagged binary payload is invalid")
    length = struct.unpack("<Q", value[:8])[0]
    payload = value[8:]
    if len(payload) != length:
        raise CanonicalEncodingError("a tagged binary payload has a bad length")
    return payload


def _require_fields(value: Mapping[str, Any], fields: set[str], label: str) -> None:
    if set(value) != fields:
        raise CanonicalEncodingError(f"the canonical {label} fields are invalid")
