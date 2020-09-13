# https://tech.davis-hansson.com/p/make/
SHELL := bash
.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c
.DELETE_ON_ERROR:
MAKEFLAGS += --warn-undefined-variables
MAKEFLAGS += --no-builtin-rules

ifeq ($(origin .RECIPEPREFIX), undefined)
  $(error This Make does not support .RECIPEPREFIX. Please use GNU Make 4.0 or later)
endif
.RECIPEPREFIX = >

# Default - top level rule is what gets ran when you run just `make`
help:
> @echo "make lint"
> @echo "	run lint checks on the codebase"
> @echo "make tests"
> @echo "	run tests"
.PHONY: help

lint:
> poetry run pre-commit run -a
.PHONY: lint

test:
> poetry run pytest tests
.PHONY: test
