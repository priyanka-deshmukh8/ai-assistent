
import ast
import math
import operator
from datetime import datetime


# Only allow these arithmetic operations.
BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

MAX_EXPRESSION_LENGTH = 200
MAX_ABS_VALUE = 1e100


def calculate(expression: str) -> dict:
    """Safely evaluate a basic arithmetic expression."""
    if not isinstance(expression, str) or not expression.strip():
        return {"error": "Expression must be a non-empty string."}

    if len(expression) > MAX_EXPRESSION_LENGTH:
        return {"error": "Expression is too long."}

    try:
        tree = ast.parse(expression, mode="eval")

        def evaluate(node):
            if isinstance(node, ast.Expression):
                return evaluate(node.body)

            if isinstance(node, ast.Constant):
                if type(node.value) not in (int, float):
                    raise ValueError("Only numbers are allowed.")
                return node.value

            if isinstance(node, ast.BinOp):
                operation = BINARY_OPERATORS.get(type(node.op))
                if operation is None:
                    raise ValueError("Unsupported arithmetic operator.")

                left = evaluate(node.left)
                right = evaluate(node.right)

                # Limit expensive exponentiation.
                if isinstance(node.op, ast.Pow):
                    if abs(right) > 1000 or abs(left) > 1e6:
                        raise ValueError("Exponentiation limit exceeded.")

                result = operation(left, right)

                if (
                    isinstance(result, complex)
                    or not math.isfinite(result)
                    or abs(result) > MAX_ABS_VALUE
                ):
                    raise ValueError("Result is outside the supported range.")

                return result

            if isinstance(node, ast.UnaryOp):
                operation = UNARY_OPERATORS.get(type(node.op))
                if operation is None:
                    raise ValueError("Unsupported unary operator.")
                return operation(evaluate(node.operand))

            raise ValueError("Only basic arithmetic is supported.")

        result = evaluate(tree)
        return {"result": result}

    except ZeroDivisionError:
        return {"error": "Division by zero is not allowed."}
    except (SyntaxError, ValueError, OverflowError, TypeError):
        return {"error": "Invalid or unsupported arithmetic expression."}


def get_current_datetime() -> dict:
    """Return the current local date and time with timezone information."""
    now = datetime.now().astimezone()

    return {
        "datetime": now.isoformat(),
        "date": now.date().isoformat(),
        "time": now.strftime("%H:%M:%S"),
        "timezone": str(now.tzinfo),
    }


# The LLM sees these definitions and decides when to request a tool.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": (
                "Calculate a basic arithmetic expression. "
                "Supports +, -, *, /, //, %, **, and parentheses."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Arithmetic expression, e.g. (125 * 48) + 10",
                    }
                },
                "required": ["expression"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_datetime",
            "description": (
                "Get the current local date, time, and timezone "
                "of the machine running the assistant."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
        },
    },
]


# Only functions registered here can be called by the LLM.
AVAILABLE_TOOLS = {
    "calculate": calculate,
    "get_current_datetime": get_current_datetime,
}
