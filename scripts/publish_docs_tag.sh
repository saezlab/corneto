#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/publish_docs_tag.sh --tag vX.Y.Z[-suffix] [--base-url URL] [--remote origin] [--python /path/to/python] [--poetry /path/to/poetry] [--keep-temp] [--dry-run]

Examples:
  scripts/publish_docs_tag.sh --tag v1.0.0-beta.3
  scripts/publish_docs_tag.sh --tag v1.0.0-beta.3 --base-url https://corneto.org
  scripts/publish_docs_tag.sh --tag v1.0.0-beta.3 --python /opt/homebrew/bin/python3.11
  scripts/publish_docs_tag.sh --tag v1.0.0-beta.3 --poetry /opt/homebrew/bin/poetry
EOF
}

tag=""
base_url="https://corneto.org"
remote="origin"
python_bin=""
poetry_bin=""
keep_temp=0
dry_run=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tag)
      tag="${2:-}"
      shift 2
      ;;
    --base-url)
      base_url="${2:-}"
      shift 2
      ;;
    --remote)
      remote="${2:-}"
      shift 2
      ;;
    --python)
      python_bin="${2:-}"
      shift 2
      ;;
    --poetry)
      poetry_bin="${2:-}"
      shift 2
      ;;
    --keep-temp)
      keep_temp=1
      shift
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "${tag}" ]]; then
  echo "Missing --tag" >&2
  usage >&2
  exit 1
fi

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

if [[ -z "${poetry_bin}" ]]; then
  if command -v poetry >/dev/null 2>&1; then
    poetry_bin="$(command -v poetry)"
  else
    echo "Poetry not found on PATH. Use --poetry /path/to/poetry" >&2
    exit 1
  fi
fi

echo "==> Fetching tags from ${remote}"
git fetch --tags "${remote}"

if ! git rev-parse -q --verify "refs/tags/${tag}" >/dev/null; then
  echo "Tag not found: ${tag}" >&2
  exit 1
fi

tmp_root="$(mktemp -d)"
tag_wt="${tmp_root}/tag-${tag}"
pages_wt="${tmp_root}/gh-pages"

cleanup() {
  if [[ "${keep_temp}" -eq 1 ]]; then
    echo "Keeping temp worktrees at ${tmp_root}"
    return
  fi
  git worktree remove -f "${tag_wt}" >/dev/null 2>&1 || true
  git worktree remove -f "${pages_wt}" >/dev/null 2>&1 || true
  rm -rf "${tmp_root}"
}
trap cleanup EXIT

echo "==> Creating worktree for tag ${tag}"
git worktree add "${tag_wt}" "refs/tags/${tag}"

echo "==> Creating worktree for gh-pages"
git fetch "${remote}" gh-pages:refs/remotes/"${remote}"/gh-pages >/dev/null 2>&1 || true
if git show-ref --verify --quiet "refs/remotes/${remote}/gh-pages"; then
  git worktree add "${pages_wt}" "${remote}/gh-pages"
  git -C "${pages_wt}" fetch "${remote}" gh-pages >/dev/null 2>&1 || true
  git -C "${pages_wt}" reset --hard "${remote}/gh-pages" >/dev/null
else
  echo "gh-pages branch not found on ${remote}" >&2
  exit 1
fi

echo "==> Building docs for ${tag} using ${poetry_bin}"
cd "${tag_wt}"
if [[ -n "${python_bin}" ]]; then
  "${poetry_bin}" env use "${python_bin}"
fi
if ! "${poetry_bin}" self show plugins | grep -q "poetry-dynamic-versioning"; then
  echo "Missing poetry-dynamic-versioning plugin. Install with:" >&2
  echo "  poetry self add \"poetry-dynamic-versioning[plugin]\"" >&2
  exit 1
fi
POETRY_DYNAMIC_VERSIONING_COMMANDS=install,build \
  "${poetry_bin}" install --with dev --extras docs

SPHINX_VERSION_MATCH="${tag}" DOCS_BASE_URL="${base_url}" \
  "${poetry_bin}" run sphinx-build -b html docs docs/_build/html

echo "==> Publishing docs to gh-pages/${tag}"
if [[ ! -d "${tag_wt}/docs/_build/html" ]]; then
  echo "Docs build output missing at docs/_build/html" >&2
  exit 1
fi
if [[ -z "$(ls -A "${tag_wt}/docs/_build/html")" ]]; then
  echo "Docs build output is empty at docs/_build/html" >&2
  exit 1
fi
mkdir -p "${pages_wt}/${tag}"
rsync -a --delete "${tag_wt}/docs/_build/html/" "${pages_wt}/${tag}/"

echo "==> Updating switcher"
"${poetry_bin}" run python "${repo_root}/scripts/generate_switcher.py" \
  --output "${pages_wt}/switcher.json" \
  --base-url "${base_url}"
"${poetry_bin}" run python "${repo_root}/scripts/patch_switcher_urls.py" \
  --root "${pages_wt}" \
  --new-url "${base_url}/switcher.json"

echo "==> Committing and pushing gh-pages changes"
cd "${pages_wt}"
if [[ -n "$(git status --porcelain)" ]]; then
  git add -A
  PRE_COMMIT_ALLOW_NO_CONFIG=1 git commit -m "docs: publish ${tag}"
  if [[ "${dry_run}" -eq 1 ]]; then
    echo "Dry run: skipping push."
    exit 0
  fi
  git fetch "${remote}" gh-pages >/dev/null 2>&1 || true
  if ! git merge --ff-only "${remote}/gh-pages" >/dev/null 2>&1; then
    echo "Remote gh-pages moved. Refusing to overwrite; re-run after updating." >&2
    exit 1
  fi
  git push "${remote}" gh-pages
else
  echo "No changes to commit."
fi
