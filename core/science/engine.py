from __future__ import annotations

import ast
import math
import operator
from typing import Any


_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

_FUNCTIONS = {
    "abs": abs,
    "cos": math.cos,
    "exp": math.exp,
    "log": math.log10,
    "ln": math.log,
    "sin": math.sin,
    "sqrt": math.sqrt,
    "tan": math.tan,
}

_NAMES = {
    "e": math.e,
    "pi": math.pi,
    "tau": math.tau,
}

_CONSTANTS = {
    "speed_of_light": {
        "symbol": "c",
        "value": 299_792_458,
        "unit": "m/s",
        "description": "Speed of light in vacuum.",
    },
    "gravitational_constant": {
        "symbol": "G",
        "value": 6.67430e-11,
        "unit": "m^3/(kg*s^2)",
        "description": "Newtonian gravitational constant.",
    },
    "planck_constant": {
        "symbol": "h",
        "value": 6.62607015e-34,
        "unit": "J*s",
        "description": "Planck constant.",
    },
    "boltzmann_constant": {
        "symbol": "k_B",
        "value": 1.380649e-23,
        "unit": "J/K",
        "description": "Boltzmann constant.",
    },
    "avogadro_constant": {
        "symbol": "N_A",
        "value": 6.02214076e23,
        "unit": "1/mol",
        "description": "Avogadro constant.",
    },
}


_UNIT_GROUPS = {
    "length": {
        "m": 1.0,
        "meter": 1.0,
        "meters": 1.0,
        "km": 1000.0,
        "kilometer": 1000.0,
        "kilometers": 1000.0,
        "cm": 0.01,
        "centimeter": 0.01,
        "centimeters": 0.01,
        "mm": 0.001,
        "millimeter": 0.001,
        "millimeters": 0.001,
        "ft": 0.3048,
        "foot": 0.3048,
        "feet": 0.3048,
        "in": 0.0254,
        "inch": 0.0254,
        "inches": 0.0254,
        "mi": 1609.344,
        "mile": 1609.344,
        "miles": 1609.344,
    },
    "mass": {
        "kg": 1.0,
        "kilogram": 1.0,
        "kilograms": 1.0,
        "g": 0.001,
        "gram": 0.001,
        "grams": 0.001,
        "mg": 1e-6,
        "milligram": 1e-6,
        "milligrams": 1e-6,
        "lb": 0.45359237,
        "pound": 0.45359237,
        "pounds": 0.45359237,
    },
    "time": {
        "s": 1.0,
        "sec": 1.0,
        "second": 1.0,
        "seconds": 1.0,
        "ms": 0.001,
        "millisecond": 0.001,
        "milliseconds": 0.001,
        "us": 0.000001,
        "microsecond": 0.000001,
        "microseconds": 0.000001,
        "min": 60.0,
        "minute": 60.0,
        "minutes": 60.0,
        "h": 3600.0,
        "hr": 3600.0,
        "hour": 3600.0,
        "hours": 3600.0,
        "day": 86400.0,
        "days": 86400.0,
    },
    "speed": {
        "m/s": 1.0,
        "mps": 1.0,
        "km/h": 1 / 3.6,
        "kph": 1 / 3.6,
        "mph": 0.44704,
    },
    "energy": {
        "j": 1.0,
        "joule": 1.0,
        "joules": 1.0,
        "kj": 1000.0,
        "kilojoule": 1000.0,
        "kilojoules": 1000.0,
        "cal": 4.184,
        "calorie": 4.184,
        "calories": 4.184,
        "kcal": 4184.0,
        "kilocalorie": 4184.0,
        "kilocalories": 4184.0,
        "ev": 1.602176634e-19,
    },
}


def _evaluate(node: ast.AST) -> float | int:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(
            node.value,
            (int, float),
        ):
            raise ValueError("Only numeric values are allowed.")
        return node.value

    if isinstance(node, ast.Name):
        if node.id not in _NAMES:
            raise ValueError(f"Unknown scientific constant: {node.id}")
        return _NAMES[node.id]

    if isinstance(node, ast.UnaryOp):
        operation = _UNARY_OPERATORS.get(type(node.op))
        if operation is None:
            raise ValueError("Unsupported unary operator.")
        return operation(_evaluate(node.operand))

    if isinstance(node, ast.BinOp):
        operation = _BINARY_OPERATORS.get(type(node.op))
        if operation is None:
            raise ValueError("Unsupported arithmetic operator.")

        left = _evaluate(node.left)
        right = _evaluate(node.right)

        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("Exponent is too large.")

        return operation(left, right)

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only approved scientific functions are allowed.")

        function = _FUNCTIONS.get(node.func.id)

        if function is None:
            raise ValueError(
                f"Unknown scientific function: {node.func.id}"
            )

        if node.keywords:
            raise ValueError("Keyword arguments are not supported.")

        return function(*[_evaluate(arg) for arg in node.args])

    raise ValueError("Unsupported scientific expression.")


def scientific_calculate(expression: str) -> dict[str, Any]:
    expression = expression.strip()

    if not expression:
        raise ValueError("Scientific expression cannot be empty.")

    if len(expression) > 300:
        raise ValueError("Scientific expression is too long.")

    try:
        tree = ast.parse(expression, mode="eval")
        result = _evaluate(tree.body)
    except (
        SyntaxError,
        ValueError,
        TypeError,
        ZeroDivisionError,
        OverflowError,
    ) as exc:
        raise ValueError(
            f"Invalid scientific expression: {exc}"
        ) from exc

    return {
        "operation": "scientific_calculation",
        "expression": expression,
        "result": result,
        "verified": True,
    }


def _normalize_unit(unit: str) -> str:
    value = (
        unit.strip()
        .lower()
        .replace("°", "")
        .replace(" ", "")
    )

    aliases = {
        "celsius": "c",
        "fahrenheit": "f",
        "kelvin": "k",
        "joule": "j",
        "joules": "j",
        "kilojoule": "kj",
        "kilojoules": "kj",
        "electronvolt": "ev",
        "electronvolts": "ev",
    }

    return aliases.get(value, value)


def _temperature_to_kelvin(value: float, unit: str) -> float:
    if unit == "c":
        return value + 273.15
    if unit == "f":
        return (value - 32.0) * 5.0 / 9.0 + 273.15
    if unit == "k":
        return value

    raise ValueError(f"Unsupported temperature unit: {unit}")


def _kelvin_to_temperature(value: float, unit: str) -> float:
    if unit == "c":
        return value - 273.15
    if unit == "f":
        return (value - 273.15) * 9.0 / 5.0 + 32.0
    if unit == "k":
        return value

    raise ValueError(f"Unsupported temperature unit: {unit}")


def convert_units(
    value: float,
    from_unit: str,
    to_unit: str,
) -> dict[str, Any]:
    source = _normalize_unit(from_unit)
    target = _normalize_unit(to_unit)

    temperature_units = {"c", "f", "k"}

    if source in temperature_units and target in temperature_units:
        kelvin = _temperature_to_kelvin(float(value), source)
        result = _kelvin_to_temperature(kelvin, target)
        group = "temperature"
    else:
        group = None

        for name, units in _UNIT_GROUPS.items():
            if source in units and target in units:
                group = name
                break

        if group is None:
            raise ValueError(
                f"Cannot convert '{from_unit}' to '{to_unit}'."
            )

        base_value = float(value) * _UNIT_GROUPS[group][source]
        result = base_value / _UNIT_GROUPS[group][target]

    return {
        "operation": "unit_conversion",
        "input": {
            "value": value,
            "unit": from_unit,
        },
        "output": {
            "value": result,
            "unit": to_unit,
        },
        "verified": True,
    }


def scientific_constant(name: str) -> dict[str, Any]:
    key = (
        name.strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )

    aliases = {
        "c": "speed_of_light",
        "speed_of_light": "speed_of_light",
        "speed_of_light_in_vacuum": "speed_of_light",
        "g": "gravitational_constant",
        "gravitational_constant": "gravitational_constant",
        "h": "planck_constant",
        "planck": "planck_constant",
        "planck_constant": "planck_constant",
        "kb": "boltzmann_constant",
        "k_b": "boltzmann_constant",
        "boltzmann_constant": "boltzmann_constant",
        "na": "avogadro_constant",
        "n_a": "avogadro_constant",
        "avogadro": "avogadro_constant",
        "avogadro_constant": "avogadro_constant",
    }

    canonical = aliases.get(key)

    if canonical is None:
        raise ValueError(f"Unknown scientific constant: {name}")

    constant = dict(_CONSTANTS[canonical])
    constant.update(
        {
            "operation": "scientific_constant",
            "name": canonical,
            "verified": True,
        }
    )

    return constant
