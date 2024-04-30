# Contributing to CORNETO

Thank you for your interest in contributing to CORNETO! We are currently in the alpha phase of development, focusing on creating and testing core functionalities, and are also in the process of preparing a manuscript that details the novel methods included in the framework. Due to these activities, the project remains closed to public contributions to maintain a controlled and focused development environment.

We value community input and look forward to opening up for contributions once we transition to a more stable beta phase. Please stay tuned for updates and feel free to provide feedback or express interest in contributing through GitHub issues.

## Setup environment

We use [Poetry](https://python-poetry.org) for dependency management. Please follow the instructions to install `poetry` on your system: https://python-poetry.org/docs/#installing-with-pipx. We recommend to install poetry using `pipx`. Once installed, clone the repository and install it with poetry. This will create a virtual environment ready for development:

```
git clone https://github.com/saezlab/corneto.git
cd corneto
poetry install
```

## Pre-commit Hooks

We use pre-commit hooks to ensure that contributions meet our coding standards and to prevent common coding issues. Pre-commit is a framework that manages and maintains multi-language pre-commit hooks.

### Using Ruff with Pre-commit

We use `ruff`, a fast, modern linter for Python, as part of our pre-commit hooks. `ruff` helps in checking the style and quality of the Python code before it is committed to the repository.

### Setting Up Pre-commit

Follow these steps to set up pre-commit on your local development environment using Poetry:

1. **Install the Git Hook Scripts**: 
    Run the following command in your repository to install the pre-commit hooks via Poetry:
    ```bash
    poetry run pre-commit install
    ```

2. **Run Pre-commit**:
    After setup, pre-commit will run automatically on git commit. However, you can manually run it on all files in the project to see if there are any issues:
    ```bash
    poetry run pre-commit run --all-files
    ```


### How It Works
When you commit changes, the pre-commit hook triggers ruff to check the staged files for style errors or other issues based on the defined rules. If ruff finds errors that it cannot automatically fix, the commit will be blocked until these errors are resolved. This helps maintain code quality and consistency across contributions.

By incorporating ruff with pre-commit into our workflow, we streamline code reviews and maintain a high standard for code quality. Please ensure you have this setup in your local development environment to aid in smooth contributions to the project.


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

We use `pytest` for running our automated tests. To run tests, use Poetry to execute the tests by running:

```bash
poetry run pytest
```

This command will run all test files in your project that follow the `test_*.py` naming convention, as recognized by `pytest`.

### Writing Tests

When writing tests, ensure each test function is clear and focused on a specific functionality. Here’s an example of a simple test:

```python
# tests/test_calculation.py

def test_calculate_division():
    from calculations import calculate_division
    assert calculate_division(10, 2) == 5.0
    with pytest.raises(ValueError):
        calculate_division(5, 0)
```

### Best Practices for Testing

- **Isolation**: Each test should be independent of others; changes in one test should not affect any other.
- **Coverage**: Aim for as much code coverage as possible to ensure that all code paths and scenarios are tested.
- **Documentation**: Document what each test covers and any specific scenarios it is testing.
- **Simplicity**: Keep tests simple and easy to understand. Complex tests can become a source of bugs themselves.
