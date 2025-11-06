# Development

## Setting Up uv

This project is set up to use [uv](https://docs.astral.sh/uv/) to manage Python and
dependencies. First, be sure you
[have uv installed](https://docs.astral.sh/uv/getting-started/installation/).

Then [fork the maedmatt/hri-playground
repo](https://github.com/maedmatt/hri-playground/fork) (having your own
fork will make it easier to contribute) and
[clone it](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository).

## Basic Developer Workflows

The `Makefile` offers shortcuts to `uv` commands for convenience.

Install dependencies:
```shell
make install
```

Run everything (sync, lint, test):
```shell
make
```

Run linting:
```shell
make lint
```

Run tests:
```shell
make test
```

Clean build artifacts:
```shell
make clean
```

Upgrade dependencies:
```shell
make upgrade
```

## Running Tests

Run all tests:
```shell
uv run pytest
```

Run specific test with output:
```shell
uv run pytest -s src/some_file.py
```

## Running Scripts

Run the main script:
```shell
uv run src/main.py
```

## Managing Dependencies

Add a dependency:
```shell
uv add package_name
```

Add a dev dependency:
```shell
uv add --dev package_name
```

Update all dependencies:
```shell
uv sync --upgrade
```

Update specific package:
```shell
uv lock --upgrade-package package_name
```

## Activating the Environment

```shell
source .venv/bin/activate
```

See [uv docs](https://docs.astral.sh/uv/) for details.

## Agent Rules

Agent rules are in [.cursor/rules](.cursor/rules). The Makefile generates `CLAUDE.md` and `AGENTS.md` from these.

Generate agent rules:
```shell
make agent-rules
```

## IDE setup

If you use VSCode or a fork like Cursor or Windsurf, you can install the following
extensions:

- [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python)

- [Based Pyright](https://marketplace.visualstudio.com/items?itemName=detachhead.basedpyright)
  for type checking. Note that this extension works with non-Microsoft VSCode forks like
  Cursor.

## Documentation

- [uv docs](https://docs.astral.sh/uv/)

- [basedpyright docs](https://docs.basedpyright.com/latest/)

* * *

*This file was built with
[simple-modern-uv](https://github.com/jlevy/simple-modern-uv).*
