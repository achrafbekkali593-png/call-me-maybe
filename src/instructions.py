"""Build the natural language text placed in front of the model.

The instructions do not have to produce valid JSON on their own: the
decoder enforces the structure. They only need to give the model enough
context to pick the right function and the right argument values.
"""

from collections.abc import Sequence

from .schema import FunctionSpec

HEADER = (
    "You are a function-calling assistant. Given the user "
    "request and the list of "
    "available functions below, choose exactly one function "
    "and its arguments. If the request does not clearly match any "
    "of the other functions, choose fn_unknown. "
    "Respond with a single JSON object of the form:\n"
)


def describe(function: FunctionSpec) -> str:
    """Return the one-line summary of a function shown to the model."""
    params = ", ".join(
        f"{key}: {spec.type.value}"
        for key, spec in function.parameters.items()
    )
    return (
        f"- {function.name} ({params}) -> "
        f"{function.returns.type.value}, {function.description}"
    )


def build_instructions(
    functions: Sequence[FunctionSpec], prompt: str
) -> str:
    """Describe the callable functions, then state the user's request."""
    catalogue = "".join(f"{describe(function)}\n" for function in functions)
    return (
        f"{HEADER}\n"
        f"available functions:\n{catalogue}\n"
        f"User request: {prompt}\n"
    )
