"""Constrained decoding: build the answer one allowed token at a time.

The model is never asked to write a whole JSON document. Instead the
program writes every character that the schema already determines (the
braces, the keys, the quotes, the commas) and only calls the model where
a real choice exists: which function to call, and what each argument
value is. Even there the choice is restricted, by keeping only the token
ids that can still lead to a valid document and picking the one the
model scores highest.
"""

import json
from collections.abc import Iterable

from llm_sdk import Small_LLM_Model
from pydantic import BaseModel, ConfigDict

from .instructions import build_instructions
from .schema import FunctionSpec, ValueType

# Tokens that either continue a number or end a JSON value.
DIGITS_OR_END = "0123456789,}"

# Safety nets: a value can never be longer than this many tokens. They
# are only reached when the model refuses to close a value, which is
# rare -- but each wasted step costs a full forward pass, so the limits
# are kept tight. 16 digits and roughly 120 characters of text are far
# more than any sane argument needs.
MAX_NUMBER_TOKENS = 16
MAX_STRING_TOKENS = 32


class ConstrainedDecoder(BaseModel):
    """Turn one natural language prompt into one JSON function call.

    The decoder keeps a single buffer of token ids, `tokens`. Fixed text
    is appended to it directly; generated text is appended one token at
    a time. When the buffer is complete, the part written after the
    instructions is decoded back into the JSON answer.

    `llm` is a plain object from the provided SDK rather than something
    pydantic can describe, hence arbitrary_types_allowed; every other
    field is validated normally.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    llm: Small_LLM_Model
    functions: list[FunctionSpec]
    tokens: list[int] = []
    cache: dict[str, list[int]] = {}

    # --- talking to the tokenizer and the model -----------------------

    def find(self, name: str) -> FunctionSpec | None:
        """Return the declared function called name, if there is one."""
        for function in self.functions:
            if function.name == name:
                return function
        return None

    def ids(self, text: str) -> list[int]:
        """Return the token ids of text, caching the fixed fragments."""
        if text not in self.cache:
            self.cache[text] = self.llm.encode(text).tolist()[0]
        return self.cache[text]

    def text_of(self, token_ids: list[int]) -> str:
        """Turn token ids back into the text they stand for."""
        return self.llm.decode(token_ids)

    def emit(self, text: str) -> None:
        """Append a piece of already decided text to the buffer."""
        self.tokens.extend(self.ids(text))

    def best_token(self, allowed: Iterable[int] | None = None) -> int:
        """Ask the model for the next token, restricted to allowed.

        Masking the forbidden logits to -inf and taking the argmax gives
        the same answer as scoring only the allowed ids and taking the
        best one, so we do the second: it looks at a handful of values
        instead of the whole 150k-entry vocabulary.

        When the constraint leaves a single possibility the model is not
        consulted at all. Its opinion could not change the outcome, and
        a forward pass is by far the most expensive thing here.
        """
        if allowed is None:
            logits = self.llm.get_logits_from_input_ids(self.tokens)
            return logits.index(max(logits))
        choices = sorted(allowed)
        if len(choices) == 1:
            return choices[0]
        logits = self.llm.get_logits_from_input_ids(self.tokens)
        return max(choices, key=lambda token: logits[token])

    # --- generating one value -----------------------------------------

    def pick_sequence(
        self, candidates: list[list[int]], stop: set[int] | None = None
    ) -> list[int]:
        """Emit one of the candidate token sequences, token by token.

        At each position only the tokens that still match at least one
        surviving candidate are allowed, so whatever comes out is always
        one of the candidates. When stop is given, those tokens may end
        the sequence, but only once some candidate has been spelled out
        in full: stopping earlier would produce a prefix of a name
        rather than a name.
        """
        chosen: list[int] = []
        position = 0
        while candidates:
            allowed = {
                candidate[position]
                for candidate in candidates
                if position < len(candidate)
            }
            if stop and any(len(c) == position for c in candidates):
                allowed |= stop
            if not allowed:
                break
            token = self.best_token(allowed)
            if stop and token in stop:
                break
            self.tokens.append(token)
            chosen.append(token)
            candidates = [
                candidate
                for candidate in candidates
                if position < len(candidate) and candidate[position] == token
            ]
            position += 1
        return chosen

    def emit_number(self, allow_decimal: bool) -> None:
        """Emit digits, plus a single decimal point when allowed.

        Generation stops as soon as the model prefers a comma or a
        closing brace, which are the two tokens that end a JSON value.
        A number that never got a decimal point is finished with ".0" so
        that it is written as a float.
        """
        digits_or_end = set(self.ids(DIGITS_OR_END))
        dot = self.ids(".")[0]
        seen_dot = False

        for _ in range(MAX_NUMBER_TOKENS):
            allowed = set(digits_or_end)
            if allow_decimal and not seen_dot:
                allowed.add(dot)
            token = self.best_token(allowed)
            piece = self.text_of([token])
            if piece.isdigit():
                self.tokens.append(token)
            elif piece == "." and allow_decimal and not seen_dot:
                self.tokens.append(token)
                seen_dot = True
            else:
                break

        if allow_decimal and not seen_dot:
            self.emit(".0")

    def emit_string(self) -> None:
        """Emit a quoted string, stopping at the first unescaped quote.

        The content itself is free: a string argument can hold anything,
        so restricting the vocabulary here would only hurt. The quotes
        around it are written by us, and a token that carries a closing
        quote is cut just before it.
        """
        self.emit('"')
        for _ in range(MAX_STRING_TOKENS):
            token = self.best_token()
            piece = self.text_of([token])
            if '"' not in piece:
                self.tokens.append(token)
                continue
            head = piece[: piece.index('"')]
            if head.endswith("\\"):
                self.tokens.append(token)
                continue
            self.emit(head)
            break
        self.emit('"')

    def emit_boolean(self) -> None:
        """Emit either true or false, narrowing between the two."""
        self.pick_sequence([self.ids("true"), self.ids("false")])

    def emit_value(self, value_type: ValueType) -> None:
        """Emit one argument value of the type the schema asks for."""
        if value_type is ValueType.STRING:
            self.emit_string()
        elif value_type is ValueType.BOOLEAN:
            self.emit_boolean()
        elif value_type is ValueType.INTEGER:
            self.emit_number(allow_decimal=False)
        else:
            self.emit_number(allow_decimal=True)

    # --- generating one complete call ---------------------------------

    def choose_function(self) -> str:
        """Let the model pick a name among the declared functions.

        The candidates are the declared names, already tokenised, so an
        invented or misspelled name is simply unreachable.
        """
        candidates = [self.ids(function.name) for function in self.functions]
        chosen = self.pick_sequence(candidates, stop=set(self.ids('"')))
        self.emit('"')
        return self.text_of(chosen)

    def emit_parameters(self, name: str) -> None:
        """Emit the parameters object declared for the chosen function."""
        self.emit(',\n"parameters" : {')
        function = self.find(name)
        parameters = function.parameters if function else {}
        for index, (key, spec) in enumerate(parameters.items()):
            if index:
                self.emit(", ")
            self.emit(f'"{key}": ')
            self.emit_value(spec.type)
        self.emit("}\n}")

    def generate(self, prompt: str) -> str:
        """Return the raw JSON text produced for a single prompt."""
        instructions = build_instructions(self.functions, prompt)
        self.tokens = self.llm.encode(instructions).tolist()[0]
        answer_starts_at = len(self.tokens)

        self.emit('{"prompt":')
        self.emit(json.dumps(prompt))
        self.emit(",\n")
        self.emit('"name":"')
        self.emit_parameters(self.choose_function())

        return self.text_of(self.tokens[answer_starts_at:])
