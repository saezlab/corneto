# Contributing to CORNETO

Thank you for your interest in contributing to CORNETO! We are currently in the alpha phase of development, focusing on creating and testing core functionalities, and are also in the process of preparing a manuscript that details the novel methods included in the framework. Due to these activities, the project remains closed to public contributions to maintain a controlled and focused development environment.

We value community input and look forward to opening up for contributions once we transition to a more stable beta phase. Please stay tuned for updates and feel free to provide feedback or express interest in contributing through GitHub issues.

## Setup environment

We use [Poetry](https://python-poetry.org) for dependency management and task execution. Please follow the instructions to install `poetry` on your system: https://python-poetry.org/docs/#installing-with-pipx. We recommend to install poetry using `pipx`.

For notebook execution in tutorials, we also support [Pixi](https://pixi.sh/) environments for isolated execution per tutorial directory.

Once installed, clone the repository and install it with poetry. This will create a virtual environment ready for development:

```bash
git clone https://github.com/saezlab/corneto.git
cd corneto
poetry install --with dev
```

## Pre-commit Hooks

We use pre-commit hooks to ensure that contributions meet our coding standards and to prevent common coding issues. These hooks are also critical for the release process, as they enforce conventional commit messages and code quality standards required for publishing (see [RELEASE.md](RELEASE.md)).

Pre-commit is a framework that manages and maintains multi-language pre-commit hooks.

### Tools Used in Our Pre-commit Setup

Our pre-commit configuration includes several tools:

- **Pre-commit Update**: Automatically keeps our pre-commit hooks up to date
- **Ruff**: A fast, modern linter and formatter for Python code
- **Conventional Commits**: Enforces conventional commit message format
- **Standard hooks**: File checks like trailing whitespace, large files, etc.
- **Poetry hooks**: Validates pyproject.toml and maintains lock file consistency

### Setting Up Pre-commit

Follow these steps to set up pre-commit on your local development environment:

1. **Install Development Dependencies**:
   First, ensure you have installed the development dependencies which include pre-commit:
   ```bash
   poetry install --with dev
   ```

2. **Install the Git Hook Scripts**:
   Install both the pre-commit and commit-msg hooks:
   ```bash
   poetry run pre-commit install --hook-type pre-commit --hook-type commit-msg
   ```

**Note**: We use an automated pre-commit-update hook that keeps our pre-commit hooks up to date automatically, so you don't need to run `pre-commit autoupdate` manually.

### Conventional Commits

We enforce conventional commit messages using the conventional-pre-commit hook. This format is essential for maintaining consistent commit history and supports the automated release process (see [RELEASE.md](RELEASE.md) for details).

Your commit messages should follow this format:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

Examples of valid commit messages:
- `feat: add new graph analysis method`
- `fix: resolve memory leak in solver backend`
- `docs: update installation instructions`
- `test: add unit tests for network import`

Common types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

### How It Works

When you commit changes, the pre-commit hooks will:

1. **Check your code** with ruff for style errors and automatically fix what it can
2. **Validate your commit message** format using conventional commits
3. **Run additional checks** like trailing whitespace removal and large file detection
4. **Verify Poetry configuration** and lock file consistency

If any hook finds errors that cannot be automatically fixed, the commit will be blocked until these errors are resolved. This helps maintain code quality and consistency across contributions.

## Code Documentation Standards

We adhere to Google's docstring style for documenting the Python code.

### Google Docstring Style

Docstrings should provide a clear explanation of the function's purpose, its arguments, what it returns, and any exceptions it raises. This includes:

- A brief description of the function.
- A detailed list of parameters and their types.
- An explanation of the return type and what the function returns.
- Any exceptions that might be raised and under what conditions.

### Example of documentation

Below is an example of a function that adheres to the Google docstring style, which `ruff` will validate for compliance:

```python
def calculate_division(numerator: float, denominator: float) -> float:
    """
    Divides the numerator by the denominator and returns the result.

    Args:
        numerator (float): The numerator of the division.
        denominator (float): The denominator of the division; must not be zero.

    Returns:
        float: The result of the division.

    Raises:
        ValueError: If the denominator is zero.

    Examples:
        >>> calculate_division(10, 2)
        5.0
        >>> calculate_division(5, 0)
        Traceback (most recent call last):
            ...
        ValueError: Denominator cannot be zero.
    """
    if denominator == 0:
        raise ValueError("Denominator cannot be zero.")
    return numerator / denominator
```

## Testing

We use `pytest` for running our automated tests. You can run tests in several ways:

### Using Poetry directly:
```bash
poetry run pytest
```

This command will run all test files in your project that follow the `test_*.py` naming convention, as recognized by `pytest`.

### Writing Tests

When writing tests, ensure each test function is clear and focused on a specific functionality. Here's an example of a simple test:

```python
# tests/test_calculation.py

def test_calculate_division():
    from calculations import calculate_division
    assert calculate_division(10, 2) == 5.0
    with pytest.raises(ValueError):
        calculate_division(5, 0)
```

## Code Quality and Testing

### Linting
```bash
poetry run ruff check corneto --exclude tests
```

### Formatting
```bash
poetry run ruff check corneto --exclude tests --fix
poetry run ruff format corneto --exclude tests
```

### Type Checking
```bash
poetry run pyrefly check corneto
```

## Generating the documentation

This project uses Sphinx along with the PyData Sphinx theme to generate HTML documentation. We also use `myst-nb` to convert Jupyter notebooks into HTML pages.

### Documentation for the current version

To generate the HTML documentation for the current version of the project using pixi:

```bash
pixi run docs
```

This command will build the documentation in the `docs/_build/html` directory, which you can open in a browser to view.

### Additional Documentation Options

We provide several pixi tasks for different documentation needs:

- **Clean build**: Remove previous builds and rebuild documentation
  ```bash
  pixi run docs-force
  ```

- **Force notebook execution**: Build docs with forced notebook execution
  ```bash
  pixi run docs-force
  ```

- **Strict mode**: Build docs treating warnings as errors
  ```bash
  pixi run python -m sphinx-build -b html -W docs docs/_build/html
  ```

- **Complete build**: Clean, force notebook execution, and build with warnings as errors
  ```bash
  pixi run docs-force
  pixi run python -m sphinx-build -b html -W docs docs/_build/html
  ```

- **Link checking**: Verify all external links in documentation
  ```bash
  pixi run python -m sphinx-build -b linkcheck docs docs/_build/linkcheck
  ```

- **Serve locally**: Build and serve documentation at http://localhost:8000
  ```bash
  pixi run docs-serve
  ```

- **Full local check**: Build HTML documentation
  ```bash
  pixi run docs
  ```

### Notebook Execution with Pixi

For tutorial notebooks, we support per-directory Pixi environments that provide isolated execution contexts. This is particularly useful when different tutorials require different dependencies or solver configurations.

Use the dedicated script to execute notebooks:
```bash
poetry run python docs/tutorials/run_notebooks.py
```

### Additional notes

- **Task automation**: Testing and code quality are run via poetry commands; documentation tasks use pixi tasks defined in `pixi.toml`.
- **`myst-nb`**: We use `myst-nb` to handle the conversion of Jupyter notebooks (`.ipynb` files) into HTML. If your contribution involves notebooks, make sure they render correctly in the generated documentation.
- **Pixi integration**: Tutorial notebooks can use individual `pixi.toml` files for isolated execution environments with specific dependencies.
- **Poetry and PEP 621**: The project uses both Poetry (legacy) and modern PEP 621 project configuration in `pyproject.toml`.

Please see [contributing with tutorials](https://github.com/saezlab/corneto/blob/main/docs/tutorials/README.md) for more information on how to contribute tutorials.

## Releases

For detailed release instructions, see [RELEASE.md](RELEASE.md).
The supported command is:

```bash
poetry run release vX.Y.Z
```

Pre-releases are also supported (for example `v1.0.0-beta.4`).
