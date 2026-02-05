# Release Process

CORNETO uses an automated tag-based release process powered by Poetry Dynamic Versioning and GitHub Actions. Git tags serve as the single source of truth for versioning - no manual version bumping in files is required.

**Note**: This document covers the release process for maintainers. For development setup and contribution guidelines, see [CONTRIBUTING.md](CONTRIBUTING.md).

## Release Workflow

To create a new release:

1. **Create and push a Git tag** following semantic versioning:
   ```bash
   git tag -a v1.2.3 -m "v1.2.3"
   git push origin v1.2.3
   ```

2. **Automatic pipeline execution**:
   - GitHub Actions detects the new tag
   - Builds the package using Poetry
   - **Creates GitHub Release with automated release notes**
   - Publishes to PyPI via OIDC trusted publishing
   - Deploys versioned documentation to GitHub Pages

3. **Version resolution**:
   - Poetry Dynamic Versioning automatically extracts the version from the Git tag
   - The package version in `pyproject.toml` remains at `0.0.0` (placeholder)
   - Built packages use the actual tag version (e.g., `1.2.3`)

4. **Automated Release Notes**:
   - GitHub automatically generates release notes based on merged PRs and commits
   - Uses conventional commit patterns to categorize changes
   - Includes contributor acknowledgments and change summaries
   - Release notes can be manually edited after creation if needed

## Example Release Process

```bash
# Ensure you're on the main branch and up to date
git checkout main
git pull origin main

# Create and push a release tag (use semantic versioning)
python scripts/release.py minor
```

The release pipeline (`.github/workflows/build-and-publish.yml`) will automatically:
- Build source and wheel distributions
- Create a GitHub Release with automated release notes
- Publish to PyPI using trusted publishing
- Deploy documentation with version switcher

### Release Helper

Use the helper script to bump and push the next tag:

```bash
python scripts/release.py major   # vX.0.0
python scripts/release.py minor   # v0.X.0
python scripts/release.py patch   # v0.0.X
```

It finds the latest `v*` tag, computes the next version, creates an annotated tag, and pushes it to `origin`.

### Customizing Release Notes

After the automated release is created, you can:
1. Go to the [GitHub Releases page](https://github.com/saezlab/corneto/releases)
2. Edit the release to add additional context, migration guides, or breaking change notices
3. The automated notes will serve as the foundation, with your manual additions

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

### Pre-commit Requirements
**Critical**: Pre-commit hooks must be installed and passing for all commits that will be included in the release. The pre-commit configuration ensures:

- **Conventional commit messages**: Required for automated GitHub release notes generation (feat:, fix:, docs:, etc.)
- **Code quality**: Linting, formatting, and style checks via Ruff
- **Repository hygiene**: File validation, trailing whitespace removal, etc.

**Important**: Conventional commits are essential for the automated release notes feature. Each commit should follow the pattern:
```
<type>(<optional scope>): <description>

[optional body]

[optional footer(s)]
```

Common types: `feat`, `fix`, `docs`, `chore`, `refactor`, `test`, `ci`, `build`, `perf`, `style`, `revert`

To set up pre-commit hooks:
```bash
poetry run pre-commit install --hook-type pre-commit --hook-type commit-msg
```

### Code Quality Validation
Before releasing, run comprehensive quality checks:
```bash
# Tests
poetry run pytest

# Linting
poetry run ruff check corneto --exclude tests

# Formatting
poetry run ruff check corneto --exclude tests --fix
poetry run ruff format corneto --exclude tests

# Type checking
poetry run pyrefly check corneto
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

### Documentation Version Switcher

The documentation uses a centralized version switcher that allows users to navigate between different documentation versions (stable, latest, and specific releases). This system is designed to work across all deployed documentation versions automatically.

#### Architecture

The version switcher uses a **single, centrally-updated file hosted on GitHub Pages**:

1. **Switcher Hosting**: `switcher.json` is deployed to the root of the docs site
   - URL: `https://corneto.org/switcher.json` (configured via `DOCS_BASE_URL`)
   - This URL is configured in `docs/conf.py` via `_switcher_url_with_ts()`
   - A timestamp parameter is added to bypass browser caching

2. **Automated Updates**: When documentation is deployed to GitHub Pages:
   - The docs workflow generates `switcher.json` (`scripts/generate_switcher.py`)
   - It is published to the site root alongside the redirect page
   - A post-deploy patch step updates existing HTML in `gh-pages` to:
     - point `theme_switcher_json_url` to `https://corneto.org/switcher.json`
     - correct `theme_switcher_version_match` based on folder (`stable/`, `latest/`, `vX.Y.Z/`)

3. **Version Mapping**: Documentation deployments use the following version identifiers:
   - `dev` branch → deployed as `"latest"`
   - `main` branch → deployed as `"stable"` (typically marked as preferred)
   - Git tags (e.g., `v1.0.0`) → deployed using the tag name

   The `SPHINX_VERSION_MATCH` environment variable (set in `.github/workflows/deploy-docs.yml`) tells the PyData Sphinx Theme which version is currently being viewed.

4. **Deployment Behavior (Why `destination_dir` is used without `keep_files`)**:
   - Each docs build replaces only its own folder (`latest/`, `stable/`, or `vX.Y.Z/`) on the `gh-pages` branch.
   - This ensures clean regeneration for a given version (no mixing of old/new files) while leaving other versions untouched.
   - The root redirect is deployed with `keep_files: true` so it doesn’t delete any versioned folders.

#### Benefits of This Approach

- **Central Management**: One `switcher.json` serves all deployed documentation versions
- **Automatic Updates**: Old deployed docs automatically show new versions in the switcher dropdown
- **No Regeneration Needed**: Previously deployed documentation doesn't need to be rebuilt to show new versions
- **Consistency**: All documentation versions display the same version list

#### Manual Switcher Updates

If the automatic workflow fails or manual intervention is needed, re-run the docs workflow or regenerate and deploy the root artifacts (redirect + `switcher.json`) using the same workflow steps.
