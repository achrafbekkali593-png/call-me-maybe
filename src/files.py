"""Read the JSON input files and write the JSON output file.

Anything that makes the run impossible (a missing file, a broken JSON
document) is raised as an InputError and reported once, at the top
level. Anything that only makes a single entry unusable is reported and
skipped, so one bad record never costs the whole run.
"""

import json
import os
from typing import Any

from pydantic import ValidationError

from .schema import FunctionCall, FunctionSpec, PromptEntry, ValueType


class InputError(Exception):
    """Raised when a file is missing, unreadable or not a JSON array."""


def read_array(path: str) -> list[Any]:
    """Read path and return the JSON array it is expected to contain."""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError:
        raise InputError(f"file not found: {path}")
    except PermissionError:
        raise InputError(f"not allowed to read {path}")
    except UnicodeDecodeError:
        raise InputError(f"{path} is not valid UTF-8 text") 
    except json.JSONDecodeError as error:
        raise InputError(f"invalid JSON in {path}: {error}")
    except OSError as error:
        raise InputError(f"could not read {path}: {error}")
    if not isinstance(data, list):
        raise InputError(
            f"expected a JSON array in {path}, got {type(data).__name__}"
        )
    return data


def is_usable(function: FunctionSpec) -> bool:
    """Say whether a definition is complete enough to be called.

    Pydantic already guaranteed the fields exist and have the right
    types. What is left to check is that none of them is blank, since a
    function with no name cannot be selected and a parameter with no
    name cannot be written into the JSON answer.
    """
    if not function.name.strip():
        return False
    if not function.description.strip():
        return False
    for key in function.parameters:
        if not key.strip():
            return False
    return True


def load_functions(path: str) -> list[FunctionSpec]:
    """Load the callable functions, skipping the unusable entries."""
    functions: list[FunctionSpec] = []
    for entry in read_array(path):
        try:
            function = FunctionSpec(**entry)
        except (ValidationError, TypeError) as error:
            print(f"Skipped a function definition: {error}")
            continue
        if not is_usable(function):
            print(
                f"Skipped '{function.name}': the name, the description "
                "or a parameter name is empty"
            )
            continue
        functions.append(function)
    return functions


def load_prompts(path: str) -> list[PromptEntry]:
    """Load the prompts to process, skipping the unusable entries."""
    prompts: list[PromptEntry] = []
    for entry in read_array(path):
        try:
            item = PromptEntry(**entry)
        except (ValidationError, TypeError) as error:
            print(f"Skipped a prompt: {error}")
            continue
        if not item.prompt.strip():
            print("Skipped an empty prompt")
            continue
        prompts.append(item)
    return prompts


def has_type(value: object, value_type: ValueType) -> bool:
    """Say whether value is an instance of the declared JSON type."""
    if value_type is ValueType.BOOLEAN:
        return isinstance(value, bool)
    if isinstance(value, bool):
        # bool is a subclass of int in Python, so it has to be ruled out
        # before any numeric check or True would pass as an integer.
        return False
    if value_type is ValueType.STRING:
        return isinstance(value, str)
    if value_type is ValueType.INTEGER:
        return isinstance(value, int)
    return isinstance(value, (int, float))


def check_call(
    call: dict[str, Any], functions: list[FunctionSpec]
) -> None:
    """Report, without raising, any departure from the output schema.

    Constrained decoding already makes a wrong answer unreachable, so
    this should never print anything. It is kept as an explicit check of
    the rules the subject lists: the three required keys, a declared
    function name, every declared argument present, no extra arguments,
    and each value of the declared type.
    """
    try:
        result = FunctionCall(**call)
    except ValidationError as error:
        for issue in error.errors():
            print(f"  schema warning: {issue['msg']}")
        return

    declared = None
    for function in functions:
        if function.name == result.name:
            declared = function
    if declared is None:
        print(f"  schema warning: '{result.name}' is not declared")
        return

    for key, spec in declared.parameters.items():
        if key not in result.parameters:
            print(f"  schema warning: argument '{key}' is missing")
        elif not has_type(result.parameters[key], spec.type):
            print(
                f"  schema warning: argument '{key}' should be "
                f"{spec.type.value}"
            )
    for key in result.parameters:
        if key not in declared.parameters:
            print(f"  schema warning: argument '{key}' is not declared")


def write_results(path: str, calls: list[dict[str, Any]]) -> None:
    """Write the calls as a JSON array, creating the folder if needed."""
    folder = os.path.dirname(path)
    try:
        if folder:
            os.makedirs(folder, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(calls, handle, indent=4)
    except OSError as error:
        raise InputError(f"could not write {path}: {error}") from None
