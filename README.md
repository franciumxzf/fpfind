# fpfind

## Quick start

```
make install
make
make test
```

## Installation

Run the `install` rule in Makefile, which will install [Poetry](https://python-poetry.org/) and the dependencies listed in [pyproject.toml](pyproject.toml).

```
make install
```

If `Make` is unavailable on your system, e.g. on Windows (although a [GNU Win32 port of Make](https://gnuwin32.sourceforge.net/packages/make.htm) is available), the commands in the Makefile can be run manually as well,

```
pip install -U poetry
poetry install
```

The commands install the `poetry` build system in the local Python installation, before installing the project dependencies and finally the library itself. This has the benefit of developing and testing within an automatically created virtualenv to avoid dependency pollution, using `poetry run`.

### Developing with Poetry

As the `fpfind` library is installed in editable mode, some notes when developing:

1. Internal importing should import from the `fpfind` library, as per usual package development workflows.
1. Testing of the library should primarily be done via a testing framework, i.e. [test-driven development](https://en.wikipedia.org/wiki/Test-driven_development).
   * Alternatively, activate the virtual environment to import `fpfind` for running locally, i.e. `poetry shell; python; import fpfind;`
   * In a one-liner: `poetry run python -ic "import fpfind"` 

Since the `poetry` package installed in this manner will tie it to the local Python installation, the `poetry.lock` should not be committed to the repository. Once onboarding instructions are changed to [system-wide installation](https://python-poetry.org/docs/), dependency locking should be enabled by removing `poetry.lock` from the `.gitignore` file.

### Local installation

If a local installation of `fpfind` is desired, use the following command:

```
pip install -e .
```

## Testing

Tests can be run using `make test`, or equivalently:

```
poetry run pytest
```

If successful, the following screen below should appear:

```
> make test
poetry run pytest
===================== test session starts =====================
platform linux -- Python 3.10.4, pytest-7.2.1, pluggy-1.0.0
rootdir: /srv/code/github/Clock-sync
collected 1 item

tests/test_py2c_example_hello.py .                      [100%]

====================== 1 passed in 0.34s ======================
```

