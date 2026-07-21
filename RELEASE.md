# Release Process

CORNETO uses an automated tag-based release process powered by Poetry Dynamic Versioning and GitHub Actions. Git tags serve as the single source of truth for versioning - no manual version bumping in files is required.

**Note**: This document covers the release process for maintainers. For development setup and contribution guidelines, see [CONTRIBUTING.md](CONTRIBUTING.md).

## Release Workflow

To create a new release:

1. **Integrate `dev` into `main`**.
   - Merge via pull request (`dev` -> `main`) or fast-forward merge locally.
   ```bash
   git checkout dev
   git pull --ff-only origin dev
   git checkout main
   git pull --ff-only origin main
   git merge --ff-only dev
   git push origin main
   ```

2. **Create and push the release tag from `main`**:
   ```bash
   poetry install
   git checkout main
   git pull --ff-only origin main
   poetry run release v1.2.3
   ```
   - Use pre-release tags as needed: `v1.2.3-alpha.0`, `v1.2.3-beta.0`, `v1.2.3-rc.0`.

3. **Automatic pipeline execution**:
   - GitHub Actions detects the new tag
   - Builds the package using Poetry
   - **Creates GitHub Release with automated release notes**
   - Publishes to PyPI via OIDC trusted publishing
   - Deploys versioned documentation to GitHub Pages

4. **Sync `main` back into `dev` after tagging**:
   - This keeps `dev` reachable from the latest release tags used by dynamic versioning.
   - Without this step, `dev` may continue reporting an older pre-release base (e.g., `1.0.0b3+...`).
   ```bash
   git checkout dev
   git pull --ff-only origin dev
   git merge --ff-only origin/main
   git push origin dev
   ```

5. **Version resolution**:
   - Poetry Dynamic Versioning automatically extracts the version from the Git tag
   - The package version in `pyproject.toml` remains at `0.0.0` (placeholder)
   - Built packages use the actual tag version (e.g., `1.2.3`)

6. **Automated Release Notes**:
   - GitHub automatically generates release notes based on merged PRs and commits
   - Uses conventional commit patterns to categorize changes
   - Includes contributor acknowledgments and change summaries
   - Release notes can be manually edited after creation if needed

## Example Release Process

```bash
# Integrate dev into main
git checkout dev
git pull --ff-only origin dev
git checkout main
git pull --ff-only origin main
git merge --ff-only dev
git push origin main

# Create and push a release tag explicitly
poetry run release v1.0.0-beta.4

# Sync main back into dev so dev sees the latest release tag ancestry
git checkout dev
git pull --ff-only origin dev
git merge --ff-only origin/main
git push origin dev
```

The release pipeline (`.github/workflows/build-and-publish.yml`) will automatically:
- Build source and wheel distributions
- Create a GitHub Release with automated release notes
- Publish to PyPI using trusted publishing
- Deploy documentation with version switcher

### Release Helper

Use the helper command to create and push an explicit tag:

```bash
poetry run release v1.2.3
poetry run release v1.0.0-beta.4
poetry run release 1.0.0-rc.1     # 'v' prefix is optional
```

It validates release preconditions (clean tree, on `main`, in sync with `origin/main`,
`origin/dev` merged into `main`, tag not already present), then creates an annotated tag and pushes it to `origin`.

Useful options:

```bash
poetry run release v1.0.0-beta.4 --dry-run  # validate only
poetry run release v1.0.0-beta.4 --yes      # skip confirmation prompt
poetry run release v1.0.0-beta.4 --remote public  # use an explicit release remote
```

The remote defaults to `origin`, which is appropriate for a normal clone of
the public CORNETO repository. In a private development workbench where
`origin` points to a private mirror, configure a separate release-only remote
for the public repository and pass it explicitly:

```bash
git remote add public https://github.com/saezlab/corneto.git
poetry run release v1.0.0-beta.8 --remote public --dry-run
poetry run release v1.0.0-beta.8 --remote public
```

Before tagging, publish selected private changes through a clean branch and a
pull request into public `dev`, followed by a release pull request from public
`dev` into public `main`. Never release a private development branch directly.

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
4. **Integrate into `main`** when ready for release (via pull request or fast-forward merge)
5. **Create release tag** on the `main` branch to trigger automated publishing
6. **Sync `main` back into `dev`** immediately after tagging

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
- Recommended integration: `dev` -> `main` for releases, then `main` -> `dev` immediately after tagging

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

#### Manual Switcher Updates

If the automatic workflow fails or manual intervention is needed, re-run the docs workflow or regenerate and deploy the root artifacts (redirect + `switcher.json`) using the same workflow steps.
