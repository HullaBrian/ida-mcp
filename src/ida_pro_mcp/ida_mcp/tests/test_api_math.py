"""Tests for eval_expr and float_convert tools."""

import math

from ..framework import (
    test,
    assert_is_list,
    assert_shape,
    assert_ok,
    assert_error,
    optional,
)
from ..api_core import eval_expr, float_convert


# ---------------------------------------------------------------------------
# eval_expr
# ---------------------------------------------------------------------------


@test()
def test_eval_expr_integer_addition():
    """eval_expr correctly evaluates simple integer addition."""
    result = eval_expr("2 + 2")
    assert_is_list(result, min_length=1)
    assert_ok(result[0], "result")
    assert result[0]["decimal"] == "4"
    assert result[0]["hexadecimal"] == "0x4"
    assert result[0]["float_value"] == 4.0


@test()
def test_eval_expr_hex_arithmetic():
    """eval_expr handles hex literals and mixed expressions."""
    result = eval_expr("0x10 + 0x20")
    assert_is_list(result, min_length=1)
    assert_ok(result[0], "result")
    assert result[0]["decimal"] == "48"
    assert result[0]["hexadecimal"] == "0x30"


@test()
def test_eval_expr_float_result():
    """eval_expr returns float fields when the result is not an integer."""
    result = eval_expr("(2.343 * 33534) - 0x7 ** 2")
    assert_is_list(result, min_length=1)
    assert_ok(result[0], "result")
    assert result[0]["decimal"] is None
    assert result[0]["hexadecimal"] is None
    expected = (2.343 * 33534) - 0x7**2
    assert abs(result[0]["float_value"] - expected) < 1e-6


@test()
def test_eval_expr_bitwise_operators():
    """eval_expr supports bitwise operators."""
    result = eval_expr("0xFF & 0x0F")
    assert_is_list(result, min_length=1)
    assert_ok(result[0], "result")
    assert result[0]["decimal"] == "15"
    assert result[0]["hexadecimal"] == "0xf"


@test()
def test_eval_expr_shift_operators():
    """eval_expr handles left/right bit shifts."""
    result = eval_expr("1 << 8")
    assert_is_list(result, min_length=1)
    assert_ok(result[0], "result")
    assert result[0]["decimal"] == "256"
    assert result[0]["hexadecimal"] == "0x100"


@test()
def test_eval_expr_batch():
    """eval_expr processes multiple expressions in one call."""
    result = eval_expr(["1 + 1", "0x10 * 2"])
    assert_is_list(result, min_length=2)
    assert result[0]["decimal"] == "2"
    assert result[1]["decimal"] == "32"


@test()
def test_eval_expr_invalid():
    """eval_expr reports an error for unparseable expressions."""
    result = eval_expr("not_a_number + 1")
    assert_is_list(result, min_length=1)
    assert result[0]["result"] is None
    assert_error(result[0])


@test()
def test_eval_expr_disallowed_call():
    """eval_expr rejects function calls — only literal expressions allowed."""
    result = eval_expr("__import__('os')")
    assert_is_list(result, min_length=1)
    assert result[0]["result"] is None
    assert_error(result[0])


@test()
def test_eval_expr_power_limit():
    """eval_expr rejects exponents that exceed the safe limit."""
    result = eval_expr("2 ** 9999")
    assert_is_list(result, min_length=1)
    assert result[0]["result"] is None
    assert_error(result[0])


# ---------------------------------------------------------------------------
# float_convert
# ---------------------------------------------------------------------------


@test()
def test_float_convert_hex_to_single():
    """float_convert decodes a 32-bit IEEE-754 hex value to float."""
    result = float_convert({"value": "0x3f800000"})
    assert_is_list(result, min_length=1)
    assert_ok(result[0], "result")
    bd = result[0]["result"]
    assert bd["float_value"] == 1.0
    assert bd["precision"] == "single"
    assert bd["sign"] == 0
    assert bd["biased_exponent"] == 127
    assert bd["mantissa_bits"] == "0" * 23
    assert bd["hexadecimal"] == "0x3f800000"


@test()
def test_float_convert_float_to_hex_single():
    """float_convert encodes 1.0 to its IEEE-754 single-precision hex."""
    result = float_convert({"value": "1.0"})
    assert_is_list(result, min_length=1)
    assert_ok(result[0], "result")
    assert result[0]["result"]["hexadecimal"] == "0x3f800000"


@test()
def test_float_convert_negative_single():
    """float_convert correctly sets sign bit for negative values."""
    result = float_convert({"value": "-1.0"})
    assert_is_list(result, min_length=1)
    assert_ok(result[0], "result")
    bd = result[0]["result"]
    assert bd["sign"] == 1
    assert bd["hexadecimal"] == "0xbf800000"


@test()
def test_float_convert_double_precision():
    """float_convert handles 64-bit double precision."""
    result = float_convert({"value": "1.0", "precision": "double"})
    assert_is_list(result, min_length=1)
    assert_ok(result[0], "result")
    bd = result[0]["result"]
    assert bd["precision"] == "double"
    assert bd["float_value"] == 1.0
    assert bd["hexadecimal"] == "0x3ff0000000000000"
    assert len(bd["mantissa_bits"]) == 52


@test()
def test_float_convert_roundtrip():
    """float_convert round-trips: float→hex→float returns the original value."""
    encode = float_convert({"value": "3.14", "precision": "single"})
    assert_ok(encode[0], "result")
    hex_val = encode[0]["result"]["hexadecimal"]

    decode = float_convert({"value": hex_val, "precision": "single"})
    assert_ok(decode[0], "result")
    import struct as _struct
    expected = _struct.unpack(">f", _struct.pack(">f", 3.14))[0]
    assert decode[0]["result"]["float_value"] == expected


@test()
def test_float_convert_invalid_precision():
    """float_convert reports an error for an unrecognised precision string."""
    result = float_convert({"value": "1.0", "precision": "half"})
    assert_is_list(result, min_length=1)
    assert result[0]["result"] is None
    assert_error(result[0])


@test()
def test_float_convert_batch():
    """float_convert processes multiple conversions in one call."""
    result = float_convert([{"value": "1.0"}, {"value": "0x40000000"}])
    assert_is_list(result, min_length=2)
    assert result[0]["result"]["float_value"] == 1.0
    assert result[1]["result"]["float_value"] == 2.0


@test()
def test_float_convert_string_input():
    """float_convert accepts a bare string via the string_parser shortcut."""
    result = float_convert("0x3f800000")
    assert_is_list(result, min_length=1)
    assert_ok(result[0], "result")
    assert result[0]["result"]["float_value"] == 1.0
