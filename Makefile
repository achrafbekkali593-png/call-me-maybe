PYTHON := uv run python

# 42 workstations give $HOME a small quota but /goinfre plenty of room,
# and torch + transformers + the model weights do not fit in the quota.
# When /goinfre/<login> exists, the uv and Hugging Face caches are moved
# there. Keeping the cache on the same filesystem as .venv also lets uv
# hardlink instead of copy. On any other machine this block does nothing.
LOGIN := $(if $(USER),$(USER),$(shell id -un))
SCRATCH := $(wildcard /goinfre/$(LOGIN))
ifneq ($(SCRATCH),)
export UV_CACHE_DIR := $(SCRATCH)/.cache/uv
export HF_HOME := $(SCRATCH)/.cache/huggingface
endif

MYPY_FLAGS := --warn-return-any --warn-unused-ignores \
	--ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

.PHONY: install run debug lint lint-strict clean cache-info

install:
	uv sync

run:
	$(PYTHON) -m src $(ARGS)

debug:
	$(PYTHON) -m pdb -m src $(ARGS)

lint:
	uv run flake8 .
	uv run mypy . $(MYPY_FLAGS)

lint-strict:
	uv run flake8 .
	uv run mypy . --strict

cache-info:
	@echo "uv cache : $(if $(UV_CACHE_DIR),$(UV_CACHE_DIR),$$HOME/.cache/uv)"
	@echo "HF cache : $(if $(HF_HOME),$(HF_HOME),$$HOME/.cache/huggingface)"

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .mypy_cache .pytest_cache

