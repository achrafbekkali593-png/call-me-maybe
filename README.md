*This project has been created as part of the 42 curriculum by ac*

# call me maybe

## Description

`call me maybe` translates natural-language prompts into structured, machine-executable
function calls. Given *"What is the sum of 2 and 3?"*, it does not answer `5`. It answers:

```json
{
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {"a": 2.0, "b": 3.0}
}
```

The point of the project is *how* that JSON is produced. Asking a 0.6B-parameter model to
write JSON and hoping it comes out valid works maybe a third of the time. Instead, this
program implements **constrained decoding** by hand on top of `Qwen/Qwen3-0.6B`: at every
generation step it inspects the model's logits and only allows the token ids that keep the
output both syntactically valid JSON *and* compliant with the schema declared in
`functions_definition.json`. Invalid output is not corrected after the fact — it is never
reachable in the first place.

No constrained-decoding library is used. The only contact with the model is through the
four public methods of `llm_sdk.Small_LLM_Model`.

## Instructions

Requirements: Python 3.12 and [`uv`](https://docs.astral.sh/uv/).

```bash
make install          # or: uv sync
make run              # or: uv run python -m src
```

On the first run the model weights are downloaded from the Hugging Face hub, which takes a
few minutes. Later runs start immediately.

### Disk space on a 42 workstation

The dependency tree pulls in torch and transformers, and the model weights are downloaded on
top of that. This does not fit in the `$HOME` quota, and `uv sync` fails with
`No space left on device (os error 28)` while unpacking `transformers`.

`make install` handles it: when `/goinfre/<login>` exists, both the uv cache and the Hugging
Face cache are redirected there. If you prefer to run `uv sync` directly, set the same two
variables first:

```bash
export UV_CACHE_DIR=/goinfre/$USER/.cache/uv
export HF_HOME=/goinfre/$USER/.cache/huggingface
```

Putting those two lines in `~/.zshrc` makes it permanent. `make cache-info` prints the paths
actually in use. If the quota is already full from a failed attempt, clear it first with
`rm -rf ~/.cache/uv ~/.cache/huggingface`.

Note that `/goinfre` is local to the machine and is wiped periodically, so the first
`make install` on a different workstation will download everything again.

Other targets:

```bash
make debug            # run under pdb
make lint             # flake8 + mypy
make lint-strict      # flake8 + mypy --strict
make clean            # remove __pycache__ and caches
```

## Example usage

Default paths (reads `data/input/`, writes `data/output/`):

```bash
uv run python -m src
```

Custom paths:

```bash
uv run python -m src \
    --functions_definition data/input/functions_definition.json \
    --input data/input/function_calling_tests.json \
    --output data/output/function_calling_results.json
```

Through the Makefile:

```bash
make run ARGS="--input data/input/my_prompts.json --output /tmp/out.json"
```

What it looks like while running:

```
[1/11] What is the sum of 2 and 3?
  -> fn_add_numbers {'a': 2.0, 'b': 3.0}
[2/11] What is the sum of 265 and 345?
  -> fn_add_numbers {'a': 265.0, 'b': 345.0}
[3/11] Greet shrek
  -> fn_greet {'name': 'shrek'}
...
Wrote 11 function call(s) to data/output/function_calling_results.json
```

## Project layout

```
src/
├── __init__.py        package docstring
├── __main__.py        entry point, turns every failure into an exit code
├── cli.py             command line options and the processing loop
├── decoder.py         the constrained decoder — the heart of the project
├── files.py           JSON reading, validation reporting, JSON writing
├── instructions.py    the natural language text given to the model
└── schema.py          pydantic models for the input and output files
```

## Algorithm explanation

### The idea

A JSON function call is mostly *not* a choice. Once the function is known, the braces, the
key names, the colons, the commas and the quotes are all fixed by the schema. Only two
things genuinely have to come from the model:

1. **which** function to call, and
2. **what** each argument value is.

So the program writes everything that is already determined itself, and calls the model
only at those two points — and even there, it restricts what the model is allowed to pick.

### The buffer

`ConstrainedDecoder` keeps one list of token ids, `self.tokens`. It starts as the tokenised
instructions. Fixed text is appended with `emit()`, which tokenises a string and extends the
buffer. Generated text is appended one token at a time. When the buffer is complete,
everything written *after* the instructions is decoded back into text — that text is the
JSON answer.

For the prompt *"Greet shrek"* the buffer grows like this:

```
<instructions>{"prompt":"Greet shrek",\n"name":"fn_greet",\n"parameters" : {"name": "shrek"}\n}
              ^--- written by us -----^^--model--^^------ written by us ------^^--model--^
```

### Restricting the model: `best_token`

Textbook constrained decoding says: take the logits, set every forbidden entry to `-inf`,
then take the argmax. Scoring only the allowed ids and keeping the best one gives exactly
the same answer, so that is what `best_token` does. Iterating over `sorted(allowed)`
reproduces the same tie-break as an argmax over a masked list (lowest token id wins), while
looking at a handful of values instead of the whole ~150 000-entry vocabulary.

Called with no argument, `best_token` does a plain unconstrained argmax — used only inside
string values, where anything is legal.

It also short-circuits: when the allowed set holds a single token there is nothing to
decide, so no forward pass is made at all.

### Choosing the function: `pick_sequence`

Every declared function name is tokenised once. At position 0 the allowed set is the first
token of every candidate. The model picks one; every candidate that does not start with
that token is dropped. Repeat at position 1, and so on.

```
candidates: fn_add_numbers  fn_greet  fn_reverse_string  fn_get_square_root
position 0: allowed = {first token of each}   -> model picks "fn"
position 1: allowed = {"_add", "_greet", "_reverse", "_get", closing quote}
                                               -> model picks "_greet"
position 2: only fn_greet survives and is exhausted
            allowed = {closing quote}          -> stop
```

The closing quote is only added to the allowed set once some candidate has been spelled out
**in full**. That detail matters: if the quote were allowed earlier, the model could stop at
any token boundary and emit `fn_substitute_string`, which is a prefix of a real name but not
a real name. Requiring a complete candidate means the model can never invent, misspell or
truncate a function name, and the loop still terminates, because once a candidate is
exhausted the closing quote is the only token left.

A useful side effect: as soon as the prefix is unambiguous, every remaining position has
exactly one allowed token. `best_token` detects that case and skips the model entirely — its
opinion could not change a forced choice, and a forward pass is by far the most expensive
operation in the program.

The same routine generates booleans, with `"true"` and `"false"` as the two candidates.

### Numbers

`emit_number` is a small state machine. The allowed set is the digit tokens plus `,` and `}`
— the two tokens that mean "this value is finished" — plus `.` when a decimal point is still
possible. If the model picks a digit it is appended; if it picks `.` the decimal flag is set
so a second one cannot follow; if it picks a terminator the loop stops *without* consuming
it, because the surrounding code writes its own separators.

A `number` that reached the end without a decimal point is finished with `.0`, which is why
the output shows `2.0` rather than `2`. An `integer` uses the same routine with decimal
points forbidden and no `.0` suffix.

### Strings

`emit_string` is deliberately the one place with no vocabulary restriction. A string
argument can hold a name, a sentence, a regex — masking would only reduce quality. The
structure is still guaranteed, because the program writes the opening and closing quotes
itself and stops at the first quote the model produces. If that quote arrives in the middle
of a multi-character token, the token is cut just before it and only the part in front is
kept. A quote preceded by a backslash is an escaped quote inside the string, so it is kept
and generation continues.

### Why the result always parses

Every character in the answer is either written by the program, or drawn from a set chosen
so the document stays well-formed. There is no path that produces a trailing comma, an
unclosed brace, an unquoted key or a name that is not in the schema. `json.loads` on the
result therefore cannot fail, and the pydantic check that follows is a safety assertion
rather than the thing that makes it work.

## Design decisions

**A class, not a pile of functions.** Every generation step needs the model, the token
buffer and the encoding cache. Holding them as attributes of `ConstrainedDecoder` keeps the
argument lists short and makes the flow readable top to bottom.

**One narrowing routine, reused.** Choosing a function name and choosing between `true` and
`false` are the same problem: pick one sequence out of a known set. `pick_sequence` solves it
once; the `stop` parameter is the only difference between the two callers.

**One number routine, reused.** `number` and `integer` differ by exactly one rule, so they
are one function with an `allow_decimal` flag rather than two near-identical copies.

**Scoring the allowed ids instead of masking the whole vocabulary.** Mathematically
identical, noticeably faster, and shorter to read. The docstring records the equivalence so
the intent is not lost.

**Encodings are cached.** `emit('"')` and the other fixed fragments are tokenised on every
value of every prompt. A dictionary keyed by the string removes a large number of redundant
tokenizer calls.

**Fixed fragments are emitted separately, never concatenated.** `,\n` and `"name":"` are two
`emit()` calls, not one. BPE can merge tokens across a boundary, so joining the strings
would change the token sequence the model actually sees.

**Declarative pydantic models, plain checks beside them.** The models in `schema.py` state
the shape of the data and let pydantic reject anything of the wrong type. The rules pydantic
cannot express — a name must not be blank — are ordinary functions in `files.py`, which keeps
both halves easy to read and easy to test. `ConstrainedDecoder` is a pydantic model too: its
`llm` field needs `arbitrary_types_allowed` because the SDK object is not something pydantic
can describe, but its function list, token buffer and cache are validated normally.

**The output is checked against the declaration, not just against a shape.** `check_call`
verifies the three required keys, that the name is one of the declared functions, that every
declared argument is present, that there are no extra ones, and that each value has the
declared type. Constrained decoding already makes a violation unreachable, so it should never
fire — it is there to make the guarantee testable rather than merely asserted. Booleans are
ruled out before the numeric checks, since `bool` is a subclass of `int` in Python and `True`
would otherwise pass as an integer.

**Fatal and non-fatal errors are separated.** A missing or corrupt file makes the whole run
impossible, so it raises `InputError` and is reported once in `__main__.py`. A single bad
record only costs that record, so it is printed and skipped. One malformed entry never
throws away the rest of the work.

## Performance analysis

**Accuracy.** Function selection is effectively exact: the model only has to distinguish
between a handful of names whose descriptions are in front of it, and wrong spellings are
impossible by construction. Argument extraction is the weaker half — the 0.6B model
occasionally copies the wrong number out of a prompt containing several, or includes a
stray word in a string. On the eleven provided test prompts the run is correct end to end.

**Validity.** 100% by construction, not by measurement. Every token is either fixed text or
drawn from a set that preserves well-formedness, so `json.loads` cannot fail on the output.

**Speed.** The cost of a run is *forward passes × sequence length*. The SDK exposes no
key/value cache, so every generated token re-runs the model over the whole sequence; keeping
both factors small is therefore the whole optimisation story.

- Only the function name and the argument values are generated. Everything else is written
  directly, so a prompt costs tens of tokens instead of hundreds.
- Forced choices cost nothing. Once a name prefix is unambiguous the rest of it is emitted
  without consulting the model. On the sample function set this removed about 80% of the
  forward passes spent selecting a name.
- The token caps on values are deliberately tight (16 and 32). They are only reached when
  the model refuses to close a value, and every wasted step is a full forward pass.
- `best_token` scores only the allowed ids instead of scanning ~150 000 logits four times
  per token, and the fixed text fragments are tokenised once and cached.

Run time is dominated by the device the SDK selects. On a GPU or Apple Silicon (fp16) the
sample set finishes in a couple of minutes; on a CPU in fp32 the same work is roughly an
order of magnitude slower, because each pass touches all 0.6B weights.

**Reliability.** No unhandled exception path is reachable from ordinary use: missing files,
unreadable files, invalid JSON, entries of the wrong shape, an empty function list, an
unwritable output folder and Ctrl-C all produce a clear message and an exit code. Unbounded
generation is prevented by hard token limits on numbers and strings.

## Challenges faced

**A quote arriving inside a larger token.** The tokenizer does not emit `"` on its own. It
regularly produces tokens like `world"` or `."`, so a naive "stop when the token is a quote"
check never fires and the string runs to its limit. The fix is to look for a quote
*anywhere* in the decoded token, keep the part in front of it, and re-encode only that part.

**Telling a closing quote from an escaped one.** A `\"` inside a string value is content,
not a terminator. Checking whether the text immediately before the quote ends in a backslash
distinguishes the two.

**Terminators must not be consumed.** The first version appended the `,` or `}` that ended a
number, and then the surrounding code wrote its own separator, producing `1.0,, `. Values
now stop *before* the terminator and let the caller write the punctuation.

**Numbers coming out as integers.** Nothing forces the model to produce a decimal point, so a
field declared `number` could be written as `2` and parse back as an `int`. Appending `.0`
when no decimal point was generated keeps the declared type.

**Concatenating fixed fragments changed the output.** Merging two `encode` calls into one
looked like a harmless cleanup, but BPE merges across the join and the model sees a different
token sequence, which changes what it predicts. The fragments are kept separate on purpose.

**Function names that were only prefixes.** Allowing the closing quote as soon as one token
had been produced let the model end a name early, so `fn_substitute_string_with_regex` could
come out as `fn_substitute_string` — a name that does not exist. The equivalence harness
caught it. The quote is now only allowed once a candidate has been spelled out in full,
which fixes the bug and, as a bonus, makes the unambiguous tail of every name free.

**Making sure a refactor changed nothing.** Described below.

## Testing strategy

**Equivalence harness.** The most useful test was a fake `Small_LLM_Model` — a char-level
tokenizer where the token id is the character's code point, and logits derived
deterministically from a hash of the current buffer. It needs no network, no torch and no
model download, it is reproducible across processes, and it exercises every branch of the
decoder. Running two versions of the code against it and diffing the output files proves
they implement the same algorithm, which is what made restructuring the project safe.

**Structural invariants.** With that harness the properties that matter can be asserted
directly rather than eyeballed: every answer parses with `json.loads`; every `name` is one of
the declared functions; every declared parameter is present with no extras; every value has
the declared Python type; and `number` fields always come back as floats.

**Edge cases on the input files.** Missing file, unreadable file, invalid JSON, a JSON object
where an array was expected, an empty array, entries that are not objects, a function with no
parameters, blank names and blank prompts. Each must produce a clear message and a clean exit
rather than a traceback.

**Edge cases on the prompts.** Empty strings, very large numbers, quotes and apostrophes
inside the request, special characters, prompts matching no function well, and functions
taking three parameters (`fn_substitute_string_with_regex` covers the last one).

**Static analysis.** `make lint` runs flake8 and mypy with the flags required by the subject;
`make lint-strict` adds `mypy --strict`. Both pass with no error.

## Resources

- [JSON specification (json.org)](https://www.json.org/)
- [Python `json` module documentation](https://docs.python.org/3/library/json.html)
- [Pydantic documentation](https://docs.pydantic.dev/)
- [Qwen3 model card (Hugging Face)](https://huggingface.co/Qwen/Qwen3-0.6B)
- [Hugging Face tokenizers — Byte-Pair Encoding](https://huggingface.co/learn/nlp-course/chapter6/5)
- [`argparse` documentation](https://docs.python.org/3/library/argparse.html)
- General background reading on grammar- and schema-constrained generation for LLMs — the
  *concept* of masking logits to force valid structured output. No constrained-decoding
  library was used; all the masking logic here is hand-written.

**How AI was used:** an AI assistant was used to understand the theory behind constrained
decoding before writing any code, to discuss how to structure the project into modules, and
to help debug the tokenizer issues described in "Challenges faced" — in particular why a
closing quote was never being detected. All the code was reviewed, tested and adjusted by
hand; no generated block was kept without understanding what it does and why.
