"""Command line options and the top level processing loop."""

import argparse
import json
import time
from typing import Any

from llm_sdk import Small_LLM_Model

from .decoder import ConstrainedDecoder
from .files import check_call, load_functions, load_prompts, write_results

DEFAULT_FUNCTIONS = "data/input/functions_definition.json"
DEFAULT_INPUT = "data/input/function_calling_tests.json"
DEFAULT_OUTPUT = "data/output/function_calling_results.json"


def parse_args() -> argparse.Namespace:
    """Read the --functions_definition, --input and --output options."""
    parser = argparse.ArgumentParser(
        prog="src",
        description=(
            "Translate natural language prompts into structured "
            "function calls."
        ),
    )
    parser.add_argument(
        "--functions_definition",
        default=DEFAULT_FUNCTIONS,
        help="JSON file describing the available functions.",
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help="JSON file containing the prompts to process.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="JSON file the resulting function calls are written to.",
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> int:
    """Process every prompt, write the results, return an exit code."""
    functions = load_functions(args.functions_definition)
    if not functions:
        print(
            "Error: no usable function definition in "
            f"{args.functions_definition}"
        )
        return 1
    prompts = load_prompts(args.input)

    print("Loading the model...")
    started = time.monotonic()
    decoder = ConstrainedDecoder(llm=Small_LLM_Model(), functions=functions)
    print(f"Model ready in {time.monotonic() - started:.1f}s")

    started = time.monotonic()
    calls: list[dict[str, Any]] = []
    for number, entry in enumerate(prompts, start=1):
        print(f"[{number}/{len(prompts)}] {entry.prompt}")
        before = time.monotonic()
        answer = decoder.generate(entry.prompt)
        elapsed = time.monotonic() - before
        try:
            call = json.loads(answer)
        except json.JSONDecodeError as error:
            print(f"  discarded, could not be parsed: {error}")
            continue
        if not isinstance(call, dict):
            print("  discarded, the answer is not a JSON object")
            continue
        check_call(call, functions)
        calls.append(call)
        print(
            f"  -> {call.get('name')} {call.get('parameters')} "
            f"[{elapsed:.1f}s]"
        )

    total = time.monotonic() - started
    write_results(args.output, calls)
    print(f"Wrote {len(calls)} function call(s) to {args.output}")
    print(f"Generation took {total:.1f}s for {len(prompts)} prompt(s)")
    return 0
