from core.science import (
    convert_units,
    scientific_calculate,
    scientific_constant,
)


def test_scientific_calculate():
    result = scientific_calculate(
        "sqrt(9) + sin(pi / 2)"
    )

    assert result["verified"] is True
    assert abs(result["result"] - 4.0) < 1e-9


def test_unit_conversion():
    result = convert_units(
        72,
        "km/h",
        "m/s",
    )

    assert result["verified"] is True
    assert abs(result["output"]["value"] - 20.0) < 1e-9


def test_scientific_constant():
    result = scientific_constant("speed of light")

    assert result["verified"] is True
    assert result["symbol"] == "c"
    assert result["value"] == 299_792_458
