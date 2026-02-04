#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/build_all_docs_local.sh [--base-url URL] [--remote origin] [--poetry /path/to/poetry] [--push]

Builds latest (dev), stable (main), and all v* tags into a local gh-pages
worktree. No push is performed. The temp directory is kept for inspection.

Examples:
  scripts/build_all_docs_local.sh
  scripts/build_all_docs_local.sh --base-url https://corneto.org
  scripts/build_all_docs_local.sh --poetry /opt/homebrew/bin/poetry
  scripts/build_all_docs_local.sh --push
EOF
}

base_url="https://corneto.org"
remote="origin"
poetry_bin=""
do_push=0
branch_name=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-url)
      base_url="${2:-}"
      shift 2
      ;;
    --remote)
      remote="${2:-}"
      shift 2
      ;;
    --poetry)
      poetry_bin="${2:-}"
      shift 2
      ;;
    --push)
      do_push=1
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

echo "==> Fetching refs from ${remote}"
git fetch --tags "${remote}"

tmp_root="$(mktemp -d)"
pages_wt="${tmp_root}/gh-pages"

cleanup() {
  if [[ "${do_push}" -eq 1 ]]; then
    git worktree remove -f "${pages_wt}" >/dev/null 2>&1 || true
    rm -rf "${tmp_root}"
  else
    echo "Keeping temp worktrees at ${tmp_root}"
  fi
}
trap cleanup EXIT

echo "==> Preparing gh-pages worktree"
git fetch "${remote}" gh-pages:refs/remotes/"${remote}"/gh-pages >/dev/null 2>&1 || true
if git show-ref --verify --quiet "refs/remotes/${remote}/gh-pages"; then
  branch_name="gh-pages-local-$(date +%Y%m%d%H%M%S)"
  git worktree add "${pages_wt}" -b "${branch_name}" "${remote}/gh-pages"
  git -C "${pages_wt}" fetch "${remote}" gh-pages >/dev/null 2>&1 || true
  git -C "${pages_wt}" reset --hard "${remote}/gh-pages" >/dev/null
else
  echo "gh-pages branch not found on ${remote}" >&2
  exit 1
fi

build_ref () {
  local ref="$1"
  local dest="$2"
  local wt="${tmp_root}/wt-${dest}"

  echo "==> Building ${dest} from ${ref}"
  git worktree add "${wt}" "${ref}"
  cd "${wt}"

  POETRY_DYNAMIC_VERSIONING_COMMANDS=install,build \
    "${poetry_bin}" install --with dev,docs

  SPHINX_VERSION_MATCH="${dest}" DOCS_BASE_URL="${base_url}" \
    "${poetry_bin}" run sphinx-build -b html docs docs/_build/html

  if [[ ! -d "docs/_build/html" ]]; then
    echo "Docs build output missing at ${wt}/docs/_build/html" >&2
    exit 1
  fi
  if [[ -z "$(ls -A docs/_build/html)" ]]; then
    echo "Docs build output is empty at ${wt}/docs/_build/html" >&2
    exit 1
  fi

  mkdir -p "${pages_wt}/${dest}"
  rsync -a --delete "docs/_build/html/" "${pages_wt}/${dest}/"

  cd "${repo_root}"
  git worktree remove -f "${wt}" >/dev/null 2>&1 || true
}

build_ref refs/heads/dev latest
build_ref refs/heads/main stable

echo "==> Building all v* tags"
for tag in $(git tag -l "v*" --sort=v:refname); do
  build_ref "refs/tags/${tag}" "${tag}"
done

echo "==> Updating root index and switcher"
cp "${repo_root}/docs/custom-index.html" "${pages_wt}/index.html"
touch "${pages_wt}/.nojekyll"
"${poetry_bin}" run python "${repo_root}/scripts/generate_switcher.py" \
  --output "${pages_wt}/switcher.json" \
  --base-url "${base_url}"
"${poetry_bin}" run python "${repo_root}/scripts/patch_switcher_urls.py" \
  --root "${pages_wt}" \
  --new-url "${base_url}/switcher.json"

echo "==> Done. Inspect local gh-pages worktree at:"
echo "${pages_wt}"
if [[ "${do_push}" -eq 1 ]]; then
  echo "==> Pushing to ${remote}/gh-pages"
  cd "${pages_wt}"
  if [[ -n "$(git status --porcelain)" ]]; then
    git add -A
    PRE_COMMIT_ALLOW_NO_CONFIG=1 git commit -m "docs: rebuild all versions"
    git push "${remote}" HEAD:gh-pages
  else
    echo "No changes to commit."
  fi
else
  echo "If you want to push later, run:"
  echo "  cd \"${pages_wt}\""
  echo "  git status -sb"
  echo "  git push ${remote} HEAD:gh-pages"
fi
