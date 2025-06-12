import json
import os
import re
import subprocess

# For robust version parsing and sorting if you have complex tags (e.g., v1.0.0-alpha.1)
# from packaging.version import parse as parse_version

# Define your package name here
PACKAGE_NAME_TO_IMPORT = "corneto"


def get_local_dynamic_version(package_name):
    """Tries to import the specified package and return its __version__."""
    try:
        module = __import__(package_name)
        version = getattr(module, "__version__", None)
        if version:
            print(f"Successfully imported '{package_name}' and found __version__: {version}")
            return version
        else:
            print(f"Warning: Imported '{package_name}' but it has no __version__ attribute.")
            return None
    except ImportError:
        print(f"Warning: Could not import package '{package_name}' to get local dynamic version.")
        return None


def get_git_tags():
    """Return a list of git tags, sorted from newest to oldest if possible."""
    try:
        subprocess.check_call(
            ["git", "fetch", "origin", "--tags", "--force"],
            text=True,
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )
        output = subprocess.check_output(["git", "tag", "--sort=-committerdate"], text=True)
        return [line.strip() for line in output.splitlines() if line.strip()]
    except subprocess.CalledProcessError as e:
        print(f"Warning: Git tag command failed. Stderr: {e.stderr}. Falling back to unsorted tags.")
        try:
            output = subprocess.check_output(["git", "tag"], text=True)
            return [line.strip() for line in output.splitlines() if line.strip()]
        except subprocess.CalledProcessError:
            print("Error: Could not retrieve git tags.")
            return []


def remote_branch_exists(branch_name):
    """Return True if the given remote branch exists (e.g. 'origin/main')."""
    try:
        subprocess.check_call(
            ["git", "remote", "update", "origin", "--prune"],
            text=True,
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )
        output = subprocess.check_output(["git", "branch", "-r"], text=True)
        return any(f"origin/{branch_name}" in line for line in output.splitlines())
    except subprocess.CalledProcessError as e:
        print(f"Error checking remote branches: {e.stderr}")
        return False


def get_deployed_versions():
    """Fetch the gh-pages branch and return a set of directory names."""
    try:
        subprocess.check_call(
            ["git", "fetch", "origin", "gh-pages:gh-pages", "--force"],
            text=True,
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )
        output = subprocess.check_output(["git", "ls-tree", "-d", "--name-only", "gh-pages"], text=True)
        deployed = {line.strip() for line in output.splitlines() if line.strip() and not line.startswith(".")}
        print(f"Found deployed versions (directories in gh-pages): {deployed}")
        return deployed
    except subprocess.CalledProcessError as e:
        print(f"Warning: Could not determine deployed versions from gh-pages. Stderr: {e.stderr}")
        return set()


def get_version_from_pyproject(git_ref, file_path="pyproject.toml"):
    """Attempts to read pyproject.toml from a git ref and extract the version."""
    try:
        output = subprocess.check_output(["git", "show", f"{git_ref}:{file_path}"], text=True, stderr=subprocess.PIPE)
        # Allow leading whitespace before the version key; escape inner quotes
        match = re.search(r"^\s*version\s*=\s*[\"']([^\"']+)[\"']", output, re.M)
        if match:
            version = match.group(1)
            print(f"Extracted version '{version}' from {file_path} on {git_ref}")
            return version
        print(f"Warning: Version pattern not found in {file_path} on {git_ref}.")
        return None
    except subprocess.CalledProcessError as e:
        print(f"Warning: 'git show {git_ref}:{file_path}' failed. Stderr: {e.stderr.strip()}")
        return None


def get_base_url():
    """Determines the base URL for documentation links."""
    url = os.environ.get("PAGE_URL")
    if url:
        print(f"Using PAGE_URL from environment: {url}")
        return url.rstrip("/")
    repo = os.environ.get("GITHUB_REPOSITORY")
    if repo:
        user, project = repo.split("/", 1)
        fallback_url = f"https://{user}.github.io/{project}"
        print(f"PAGE_URL not set. Using fallback GitHub Pages URL: {fallback_url}")
        return fallback_url
    print("Error: Could not determine base URL. GITHUB_REPOSITORY and PAGE_URL are not set.")
    return "https://your-username.github.io/your-repo"


def main():
    base_url = get_base_url()
    switcher = []
    processed_identifiers = set()

    # Get the current CI ref (branch or tag name) and the local dynamic version
    current_ci_ref_name = os.environ.get("GITHUB_REF_NAME")
    print(f"Current CI REF_NAME: {current_ci_ref_name}")
    local_dynamic_version = get_local_dynamic_version(PACKAGE_NAME_TO_IMPORT)

    branches_to_process = [
        {"id": "main", "display_name": "latest", "preferred": True},
        {"id": "dev", "display_name": "dev"},
    ]

    for branch_info in branches_to_process:
        branch_id = branch_info["id"]
        git_ref_for_remote_lookup = f"origin/{branch_id}"
        switcher_version_str = branch_id

        if branch_id == current_ci_ref_name and local_dynamic_version:
            switcher_version_str = local_dynamic_version
            print(
                f"Using local dynamic version '{switcher_version_str}' for '{branch_id}' switcher entry (CI is on '{current_ci_ref_name}')."
            )
        else:
            version_from_toml = get_version_from_pyproject(git_ref_for_remote_lookup)
            if version_from_toml and version_from_toml != "0.0.0":
                switcher_version_str = version_from_toml
            print(f"Using version '{switcher_version_str}' for '{branch_id}' switcher entry.")

        if remote_branch_exists(branch_id):
            entry = {
                "name": branch_info["display_name"],
                "version": switcher_version_str,
                "url": f"{base_url}/{branch_id}/",
            }
            if branch_info.get("preferred"):
                entry["preferred"] = True
            switcher.append(entry)
            processed_identifiers.add(branch_id)
        else:
            print(f"Remote branch 'origin/{branch_id}' not found. Skipping.")

    deployed = get_deployed_versions()
    if not deployed and os.environ.get("GITHUB_ACTIONS") == "true":
        print("Warning: Could not get deployed versions. Tags will be added without checking deployment status.")

    all_tags = get_git_tags()
    print(f"Found git tags: {all_tags}")

    for tag in all_tags:
        if tag in processed_identifiers:
            continue
        if deployed and tag not in deployed:
            continue

        tag_version_str = tag
        if tag == current_ci_ref_name and local_dynamic_version:
            tag_version_str = local_dynamic_version
            print(f"CI is building tag '{tag}'. Using local dynamic version '{local_dynamic_version}'.")

        entry = {"name": tag, "version": tag_version_str, "url": f"{base_url}/{tag}/"}
        switcher.append(entry)
        processed_identifiers.add(tag)

    final_switcher = []
    main_entry = next((e for e in switcher if e.get("preferred")), None)
    dev_entry = next((e for e in switcher if e["name"] == "dev" and not e.get("preferred")), None)
    if main_entry:
        final_switcher.append(main_entry)
    if dev_entry:
        final_switcher.append(dev_entry)

    other_entries = [e for e in switcher if e not in (main_entry, dev_entry)]
    # For true SemVer sorting, uncomment below and import parse_version
    # other_entries.sort(key=lambda x: parse_version(x['version']), reverse=True)
    other_entries.sort(key=lambda x: x["version"], reverse=True)
    final_switcher.extend(other_entries)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, "switcher.json")

    with open(output_path, "w") as f:
        json.dump(final_switcher, f, indent=2)

    print(f"Switcher file generated at '{output_path}' with {len(final_switcher)} entries.")


if __name__ == "__main__":
    main()
