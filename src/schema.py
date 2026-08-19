"""Pydantic models describing the input files and the output file.

The models are purely declarative: pydantic checks that every field is
present and has the right type, and refuses anything else. The few extra
rules pydantic cannot express on its own (a name must not be blank, for
instance) live in files.py, next to the code that reports them.
"""

from enum import StrEnum

from pydantic import BaseModel


class ValueType(StrEnum):
    """JSON value types a parameter or a return value may take."""

    NUMBER = "number"
    INTEGER = "integer"
    STRING = "string"
    BOOLEAN = "boolean"


class ValueSpec(BaseModel):
    """Declared type of a single parameter or of a return value."""

    type: ValueType


class FunctionSpec(BaseModel):
    """One callable function, as declared in functions_definition.json."""

    name: str
    description: str
    parameters: dict[str, ValueSpec]
    returns: ValueSpec


class PromptEntry(BaseModel):
    """One natural language request read from the input file."""

    prompt: str


class FunctionCall(BaseModel):
    """One result written to the output file."""

    prompt: str
    name: str
    parameters: dict[str, int | float | str | bool]
