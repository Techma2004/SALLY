from __future__ import annotations

import ast
import operator


_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def calculate(expression: str) -> str:
    expression = expression.strip()

    if not expression:
        raise ValueError("Expression cannot be empty.")

    if len(expression) > 200:
        raise ValueError("Expression is too long.")

    try:
        tree = ast.parse(expression, mode="eval")
        result = _evaluate(tree.body)
    except (SyntaxError, ValueError, TypeError, ZeroDivisionError, OverflowError) as exc:
        raise ValueError(f"Invalid arithmetic expression: {exc}") from exc

    return str(result)


def _evaluate(node: ast.AST):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(
            node.value,
            (int, float),
        ):
            raise ValueError("Only numeric values are allowed.")
        return node.value

    if isinstance(node, ast.UnaryOp):
        operation = _OPERATORS.get(type(node.op))
        if operation is None:
            raise ValueError("Unsupported unary operator.")
        return operation(_evaluate(node.operand))

    if isinstance(node, ast.BinOp):
        operation = _OPERATORS.get(type(node.op))
        if operation is None:
            raise ValueError("Unsupported operator.")

        left = _evaluate(node.left)
        right = _evaluate(node.right)

        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("Exponent is too large.")

        return operation(left, right)

    raise ValueError("Only basic arithmetic is supported.")



def _natural_number(value: str) -> str:
    return value.strip().replace(",", "")


def natural_expression(
    text: str,
    *,
    base_value: str | None = None,
) -> str | None:
    """
    Convert common natural-language arithmetic into a safe arithmetic
    expression. Returns None when the phrase is not recognized.
    """
    import re

    text = text.strip().rstrip("?.!")
    lowered = text.lower()

    if re.fullmatch(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", lowered):
        return lowered

    # Follow-up operations such as "divide by 56".
    if base_value is not None:
        follow = re.fullmatch(
            r"(?:please\s+)?"
            r"(add|subtract|multiply|divide)"
            r"(?:\s+the\s+result)?"
            r"(?:\s+by|\s+with|\s+and)?\s*"
            r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:\s*[\+\-\*/%]\s*"
            r"[-+]?(?:\d+(?:\.\d*)?|\.\d+))*)",
            lowered,
        )
        if follow:
            operator_name, operand = follow.groups()
            operator_symbol = {
                "add": "+",
                "subtract": "-",
                "multiply": "*",
                "divide": "/",
            }[operator_name]
            return f"({base_value}) {operator_symbol} ({operand})"

    # Explicit binary phrases.
    patterns = (
        (
            r"(?:could you\s+)?multiply\s+"
            r"([-+]?(?:\d+(?:\.\d*)?|\.\d+))\s+by\s+"
            r"([-+]?(?:\d+(?:\.\d*)?|\.\d+))",
            "*",
        ),
        (
            r"([-+]?(?:\d+(?:\.\d*)?|\.\d+))\s+times\s+"
            r"([-+]?(?:\d+(?:\.\d*)?|\.\d+))",
            "*",
        ),
        (
            r"(?:add)\s+"
            r"([-+]?(?:\d+(?:\.\d*)?|\.\d+))\s+and\s+"
            r"([-+]?(?:\d+(?:\.\d*)?|\.\d+))",
            "+",
        ),
        (
            r"(?:subtract)\s+"
            r"([-+]?(?:\d+(?:\.\d*)?|\.\d+))\s+from\s+"
            r"([-+]?(?:\d+(?:\.\d*)?|\.\d+))",
            "reverse_subtract",
        ),
        (
            r"([-+]?(?:\d+(?:\.\d*)?|\.\d+))\s+minus\s+"
            r"([-+]?(?:\d+(?:\.\d*)?|\.\d+))",
            "-",
        ),
        (
            r"(?:divide)\s+"
            r"([-+]?(?:\d+(?:\.\d*)?|\.\d+))\s+by\s+"
            r"([-+]?(?:\d+(?:\.\d*)?|\.\d+))",
            "/",
        ),
        (
            r"([-+]?(?:\d+(?:\.\d*)?|\.\d+))\s+divided\s+by\s+"
            r"([-+]?(?:\d+(?:\.\d*)?|\.\d+))",
            "/",
        ),
        (
            r"([-+]?(?:\d+(?:\.\d*)?|\.\d+))\s+plus\s+"
            r"([-+]?(?:\d+(?:\.\d*)?|\.\d+))",
            "+",
        ),
    )

    for pattern, operator_symbol in patterns:
        match = re.fullmatch(pattern, lowered)
        if not match:
            continue

        left, right = match.groups()

        if operator_symbol == "reverse_subtract":
            return f"({_natural_number(right)}) - ({_natural_number(left)})"

        return (
            f"({_natural_number(left)}) "
            f"{operator_symbol} "
            f"({_natural_number(right)})"
        )

    # Chained operations:
    # "multiply 45 by 67 and add with 100 and divide by 100 ..."
    parts = re.split(r"\s+\band\b\s+", lowered)

    if len(parts) >= 2:
        first = parts[0].strip()
        base_expression = natural_expression(
            first,
            base_value=base_value,
        )

        if base_expression:
            expression = base_expression

            for part in parts[1:]:
                part = part.strip()
                match = re.fullmatch(
                    r"(add|subtract|multiply|divide)"
                    r"(?:\s+with|\s+by|\s+and)?\s*"
                    r"(.+)",
                    part,
                )

                if not match:
                    return None

                operation, operand = match.groups()
                if not re.fullmatch(
                    r"[-+*/%().\d\s]+",
                    operand,
                ):
                    return None

                symbol = {
                    "add": "+",
                    "subtract": "-",
                    "multiply": "*",
                    "divide": "/",
                }[operation]

                expression = (
                    f"({expression}) {symbol} ({operand.strip()})"
                )

            return expression

    return None
