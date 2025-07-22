# Contributing to Documentation

This guide explains how to build and work with the CORNETO documentation locally.

## Environment Variables

### SPHINX_VERSION_MATCH

The `SPHINX_VERSION_MATCH` environment variable controls version display in the documentation:

- **Purpose**: Ensures version switcher and content version display align with deployment folders
- **In CI**: Automatically set to deployment folder name (`main`, `dev`, `v1.0.0`, etc.)
- **Locally**: Defaults to `corneto.__version__` if not set

**Usage examples:**

```bash
# Build docs with version showing as "main"
SPHINX_VERSION_MATCH=main sphinx-build -b html docs docs/_build/html

# Build docs with version showing as "dev"
SPHINX_VERSION_MATCH=dev sphinx-build -b html docs docs/_build/html

# Build docs with version showing as a tag
SPHINX_VERSION_MATCH=v1.0.0 sphinx-build -b html docs docs/_build/html

# Default behavior (uses package version)
sphinx-build -b html docs docs/_build/html
```

**What it affects:**
- Version switcher `version_match` (which version is highlighted as current)
- `{{version}}` substitutions in Markdown files (e.g., in `docs/index.md`)

## Local Development

### Building Documentation

1. **Install dependencies:**
   ```bash
   poetry install --with dev,docs
   ```

2. **Generate switcher.json:**
   ```bash
   poetry run python docs/switcher.py
   ```

3. **Build documentation:**
   ```bash
   # With custom version display
   SPHINX_VERSION_MATCH=dev poetry run sphinx-build -b html docs docs/_build/html

   # Or with package version (default)
   poetry run sphinx-build -b html docs docs/_build/html
   ```

4. **Serve locally:**
   ```bash
   cd docs/_build/html
   python -m http.server 8000
   ```

   Open `http://localhost:8000` in your browser.

### Testing Version Switching

To test the version switcher locally:

1. Build docs for different "versions":
   ```bash
   SPHINX_VERSION_MATCH=main poetry run sphinx-build -b html docs docs/_build/html/main
   SPHINX_VERSION_MATCH=dev poetry run sphinx-build -b html docs docs/_build/html/dev
   SPHINX_VERSION_MATCH=v1.0.0 poetry run sphinx-build -b html docs docs/_build/html/v1.0.0
   ```

2. Copy `switcher.json` to the root:
   ```bash
   cp docs/switcher.json docs/_build/html/
   ```

3. Serve and test switching between versions.

## CI/CD Integration

The documentation is automatically built and deployed by GitHub Actions:

- **Automatic**: On push to `main`, `dev`, or tags
- **Manual**: Use "Manual Deploy Docs" workflow with tag input

The CI sets `SPHINX_VERSION_MATCH` automatically to match the deployment folder structure.
