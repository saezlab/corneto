# Contributing to CORNETO

Thank you for your interest in contributing to CORNETO! We are currently in the alpha phase of development, focusing on creating and testing core functionalities, and are also in the process of preparing a manuscript that details the novel methods included in the framework. Due to these activities, the project remains closed to public contributions to maintain a controlled and focused development environment.

We value community input and look forward to opening up for contributions once we transition to a more stable beta phase. Please stay tuned for updates and feel free to provide feedback or express interest in contributing through GitHub issues.

## Setup environment

We use [Poetry](https://python-poetry.org) for dependency management and [Nox](https://nox.thea.codes/) for task automation. Please follow the instructions to install `poetry` on your system: https://python-poetry.org/docs/#installing-with-pipx. We recommend to install poetry using `pipx`.

For notebook execution in tutorials, we also support [Pixi](https://pixi.sh/) environments for isolated execution per tutorial directory.

Once installed, clone the repository and install it with poetry. This will create a virtual environment ready for development:

```bash
git clone git+https://github.com/saezlab/corneto.git@dev
cd corneto
poetry install --with dev
```

## Pre-commit Hooks

We use pre-commit hooks to ensure that contributions meet our coding standards and to prevent common coding issues. Pre-commit is a framework that manages and maintains multi-language pre-commit hooks.

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

We enforce conventional commit messages using the conventional-pre-commit hook. Your commit messages should follow this format:

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

### Using Nox (recommended):
```bash
nox -s tests
```

### Using Tox (legacy):
```bash
tox -e py
```

The nox approach is recommended as it creates an isolated environment and ensures consistent testing across different setups. Nox is our primary task runner that provides standardized environments for testing, linting, formatting, and documentation building.

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

## Code Quality and Testing with Nox

We use [Nox](https://nox.thea.codes/) to standardize testing, linting, and documentation building across different environments. Nox provides isolated virtual environments for running different tasks and supports multiple Python versions.

### Available Nox Sessions

- **Testing**: Run unit tests across supported Python versions (3.10, 3.11, 3.12)
  ```bash
  nox -s tests  # Uses your current Python version
  # Or specify a version if you have multiple:
  nox -s "tests-3.10"  # Python 3.10
  nox -s "tests-3.11"  # Python 3.11
  nox -s "tests-3.12"  # Python 3.12
  ```

- **Linting**: Check code style and quality with ruff
  ```bash
  nox -s lint
  ```

- **Formatting**: Auto-fix code formatting issues
  ```bash
  nox -s format
  ```

- **Type Checking**: Run static type analysis with mypy
  ```bash
  nox -s typing
  ```

- **Documentation**: Build HTML documentation
  ```bash
  nox -s docs
  ```

- **Notebook Caching**: Execute notebooks with Pixi isolation (for tutorials)
  ```bash
  nox -s cache_notebooks_with_pixi
  ```

### Running All Quality Checks

To run all quality checks (linting, formatting, typing, and tests) at once:
```bash
nox -s lint format typing tests
```

### Legacy Tox Support

We still maintain tox configuration for backwards compatibility:
```bash
tox -e lint,format,typing,py
```

## Generating the documentation

This project uses Sphinx along with the PyData Sphinx theme to generate HTML documentation. We also use `myst-nb` to convert Jupyter notebooks into HTML pages.

### Documentation for the current version

To generate the HTML documentation for the current version of the project using nox:

```bash
nox -s docs
```

This command will build the documentation in the `docs/_build/html` directory, which you can open in a browser to view.

### Additional Documentation Options

We provide several nox sessions for different documentation needs:

- **Clean build**: Remove previous builds and rebuild documentation
  ```bash
  nox -s docs_clean
  ```

- **Force notebook execution**: Build docs with forced notebook execution
  ```bash
  nox -s docs_force
  ```

- **Strict mode**: Build docs treating warnings as errors
  ```bash
  nox -s docs_werror
  ```

- **Complete build**: Clean, force notebook execution, and build with warnings as errors
  ```bash
  nox -s docs_all
  ```

- **Link checking**: Verify all external links in documentation
  ```bash
  nox -s docs_linkcheck
  ```

- **Serve locally**: Build and serve documentation at http://localhost:8000
  ```bash
  nox -s docs_serve
  ```

- **Generate switcher**: Generate version switcher JSON for Read-the-Docs
  ```bash
  nox -s generate_switcher
  ```

- **Full local check**: Build docs and generate switcher
  ```bash
  nox -s docs_full
  ```

### Legacy Tox Support

You can still use tox for documentation tasks:
```bash
tox -e docs
tox -e docs-clean
tox -e docs-force
tox -e docs-werror
tox -e docs-linkcheck
tox -e docs-serve
```

### Notebook Execution with Pixi

For tutorial notebooks, we support per-directory Pixi environments that provide isolated execution contexts. This is particularly useful when different tutorials require different dependencies or solver configurations.

To execute notebooks with Pixi isolation:
```bash
nox -s cache_notebooks_with_pixi
```

You can also filter specific notebooks using patterns:
```bash
nox -s cache_notebooks_with_pixi -- "*metabolic*" "^docs/.*intro.ipynb$"
```

### Additional notes

- **Task automation**: All documentation, testing, and quality assurance tasks are standardized through nox sessions defined in `noxfile.py`.
- **Legacy support**: Tox environments are still maintained in `tox.ini` for backwards compatibility.
- **`myst-nb`**: We use `myst-nb` to handle the conversion of Jupyter notebooks (`.ipynb` files) into HTML. If your contribution involves notebooks, make sure they render correctly in the generated documentation.
- **Pixi integration**: Tutorial notebooks can use individual `pixi.toml` files for isolated execution environments with specific dependencies.
- **Poetry and PEP 621**: The project uses both Poetry (legacy) and modern PEP 621 project configuration in `pyproject.toml`.

Please see [contributing with tutorials](https://github.com/saezlab/corneto/blob/dev/docs/tutorials/README.md) for more information on how to contribute tutorials.

## Releases

CORNETO uses an automated tag-based release process powered by Poetry Dynamic Versioning and GitHub Actions. Git tags serve as the single source of truth for versioning - no manual version bumping in files is required.

### Release Workflow

To create a new release:

1. **Create and push a Git tag** following semantic versioning:
   ```bash
   git tag v1.2.3
   git push origin v1.2.3
   ```

2. **Automatic pipeline execution**:
   - GitHub Actions detects the new tag
   - Builds the package using Poetry
   - Publishes to PyPI via OIDC trusted publishing
   - Deploys versioned documentation to GitHub Pages

3. **Version resolution**:
   - Poetry Dynamic Versioning automatically extracts the version from the Git tag
   - The package version in `pyproject.toml` remains at `0.0.0` (placeholder)
   - Built packages use the actual tag version (e.g., `1.2.3`)

### Example Release Process

```bash
# Ensure you're on the main branch and up to date
git checkout main
git pull origin main

# Create a release tag (use semantic versioning)
git tag v0.2.0

# Push the tag to trigger the release pipeline
git push origin v0.2.0
```

The release pipeline (`.github/workflows/build-and-publish.yml`) will automatically:
- Build source and wheel distributions
- Publish to PyPI using trusted publishing
- Deploy documentation with version switcher

### Version Numbering

Follow [Semantic Versioning](https://semver.org/):
- `MAJOR.MINOR.PATCH` (e.g., `v1.2.3`)
- Use `v` prefix for tags (e.g., `v1.0.0`, not `1.0.0`)
- Pre-releases: `v1.0.0-alpha.0`, `v1.0.0-beta.0`, `v1.0.0-rc.0`
