# Release Process

CORNETO uses public `main` as its trunk. Every public change reaches it through a
pull request, and an annotated Git tag is the single source of truth for a
release version. Poetry Dynamic Versioning reads that tag; do not edit a version
in project files for a release.

## Public workflow

1. Develop and validate changes in the private workbench as needed.
2. Review the exact publishable snapshot for confidential material.
3. Push that snapshot to a temporary public `publication/<topic>` branch and
   open one pull request into public `main`.
4. The public-main pull request is the release gate. Its CI validates the merge
   result with the Python test matrix, quality checks, documentation build, and
   package smoke test.
5. After the PR merges, fast-forward the private mirror's `main` branch to
   public `main`.
6. Create the release tag from that public `main` commit.

There is no public `dev` branch in this workflow. The private `dev` branch may
remain an unpublished integration line, but it is never a public promotion
target.

## Creating a release

Only tag a clean local checkout on `main` after the public-main PR has passed
and merged:

```bash
git checkout main
git pull --ff-only origin main
poetry install
poetry run release v1.2.3 --dry-run
poetry run release v1.2.3
```

Pre-release tags use the same flow, for example `v1.2.3-alpha.0`,
`v1.2.3-beta.0`, or `v1.2.3-rc.0`. The helper confirms that the tree is clean,
the checkout is on and matches the selected remote's `main`, and the tag does
not already exist.

Useful options:

```bash
poetry run release v1.0.0-rc.1 --dry-run
poetry run release v1.0.0-rc.1 --yes
```

In the private workbench, the public repository is configured as `public` and
must be named explicitly:

```bash
poetry run release v1.0.0-rc.1 --remote public --dry-run
poetry run release v1.0.0-rc.1 --remote public
```

## CI responsibilities

The release process intentionally gives each workflow one job:

- Pull requests into `main` run the complete release gate.
- Pushes to `main` deploy the stable documentation only.
- Release tags build the exact tagged distributions, check them, publish to
  PyPI, create the GitHub Release, and deploy versioned documentation.

This avoids repeating the full test matrix for temporary branch pushes, the
post-merge `main` push, and the release tag. The tagged package remains
protected by artifact-specific validation, while the merged PR is the code
quality gate.

## Local validation

Before opening the public-main PR, run the checks relevant to the change. A
full release-candidate check normally includes:

```bash
poetry run pytest
poetry run ruff check --no-fix corneto tests
poetry run ruff format --check corneto tests
poetry check
poetry run sphinx-build -W --keep-going -b html docs docs/_build/html
poetry build
poetry run twine check dist/*
```

The GitHub release pipeline uses OIDC trusted publishing, creates release notes
automatically, and marks alpha, beta, and RC tags as pre-releases.
