# Release Process

CORNETO uses an automated tag-based release process powered by Poetry Dynamic Versioning and GitHub Actions. Git tags serve as the single source of truth for versioning - no manual version bumping in files is required.

**Note**: This document covers the release process for maintainers. For development setup and contribution guidelines, see [CONTRIBUTING.md](CONTRIBUTING.md).

## Release Workflow

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

## Example Release Process

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

## Version Numbering

Follow [Semantic Versioning](https://semver.org/):
- `MAJOR.MINOR.PATCH` (e.g., `v1.2.3`)
- Use `v` prefix for tags (e.g., `v1.0.0`, not `1.0.0`)
- Pre-releases: `v1.0.0-alpha.0`, `v1.0.0-beta.0`, `v1.0.0-rc.0`

## Prerequisites for Releases

Before creating a release, ensure the development environment and code quality standards are met:

### Development Setup
All maintainers should have the development environment properly configured as described in [CONTRIBUTING.md](CONTRIBUTING.md), including:
- Poetry for dependency management
- Pre-commit hooks installed and active
- Nox for running quality checks

### Pre-commit Requirements
**Critical**: Pre-commit hooks must be installed and passing for all commits that will be included in the release. The pre-commit configuration ensures:

- **Conventional commit messages**: Required for consistent release notes and potential automated changelog generation
- **Code quality**: Linting, formatting, and style checks via Ruff
- **Repository hygiene**: File validation, trailing whitespace removal, etc.

To set up pre-commit hooks:
```bash
poetry run pre-commit install --hook-type pre-commit --hook-type commit-msg
```

### Code Quality Validation
Before releasing, run comprehensive quality checks:
```bash
# Run all quality checks
nox -s lint format typing tests

# Or individual checks
nox -s tests          # Run test suite
nox -s lint           # Check code style
nox -s format         # Auto-fix formatting
nox -s typing         # Type checking
```

## Development Workflow

The typical development workflow involves:

1. **Work in the `dev` branch** for ongoing development
2. **Ensure all commits follow conventional commit format** (enforced by pre-commit hooks)
3. **Run quality checks** before merging to main
4. **Merge to `main`** when ready for release (via pull request)
5. **Create release tag** on the `main` branch to trigger automated publishing

## Technical Details

### Poetry Dynamic Versioning Configuration

The project uses Poetry Dynamic Versioning (configured in `pyproject.toml`):

```toml
[tool.poetry-dynamic-versioning]
enable           = true
vcs              = "git"
pattern          = "default"
style            = "pep440"
tagged-metadata  = false
```

### GitHub Actions Workflow

The release workflow (`.github/workflows/build-and-publish.yml`) is triggered by:
- Push events to tags matching `v*` pattern
- Uses OIDC trusted publishing for secure PyPI uploads
- Requires no manual secrets or tokens

### Branch Strategy

- **`main`**: Stable releases and release tags
- **`dev`**: Active development branch
- Pull requests: `dev` → `main` for releases
