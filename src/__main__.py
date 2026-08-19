"""Program entry point, reached with `uv run python -m src`."""

import sys

from .cli import parse_args, run
from .files import InputError


def main() -> int:
    """Run the program, turning every failure into a clear message."""
    try:
        return run(parse_args())
    except InputError as error:
        print(f"Error: {error}")
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted by the user.")
        return 130
    except Exception as error:
        print(f"Unexpected error: {type(error).__name__}: {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
