#!/usr/bin/env python3
"""Generate version switcher for Sphinx documentation with PyData theme.

This script handles dynamic versioning where:
- main branch: Should show the latest stable version (e.g., "1.0.0")
- dev branch: Should show development versions (e.g., "1.0.1.dev0")
- tags: Show their specific versions

The script matches the version strings with deployed directories in gh-pages
to ensure the switcher URLs work correctly.
"""

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


def get_version_from_git_ref(git_ref):
    """Try to get the actual version from a git ref by:
    1. First trying to check out the ref and import the package
    2. If that fails, look for version in pyproject.toml
    3. If version is "0.0.0", return None to indicate dynamic versioning
    """
    # Save current branch/commit
    try:
        current_ref = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except subprocess.CalledProcessError:
        current_ref = None

    try:
        # Try to get version from pyproject.toml first
        output = subprocess.check_output(
            ["git", "show", f"{git_ref}:pyproject.toml"],
            text=True,
            stderr=subprocess.PIPE,
        )
        match = re.search(r"^\s*version\s*=\s*[\"']([^\"']+)[\"']", output, re.M)
        if match:
            version = match.group(1)
            if version != "0.0.0":
                print(f"Found static version '{version}' in pyproject.toml for {git_ref}")
                return version

        # If we get here, it's using dynamic versioning
        print(f"Detected dynamic versioning (0.0.0) for {git_ref}")

        # For dynamic versioning, we need to get the version from the actual deployed docs
        # The version should match the directory name in gh-pages
        return None

    except subprocess.CalledProcessError as e:
        print(f"Warning: Could not read pyproject.toml from {git_ref}: {e.stderr.strip()}")
        return None
    finally:
        # Restore original ref if we changed it
        if current_ref:
            try:
                subprocess.check_call(
                    ["git", "checkout", current_ref],
                    stderr=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                )
            except subprocess.CalledProcessError:
                pass


def normalize_version_for_comparison(version):
    """Normalize version string for comparison with deployed directories."""
    # Remove 'v' prefix if present
    if version.startswith("v"):
        version = version[1:]
    return version


def get_expected_version_pattern(branch_name, current_version=None):
    """Determine what version pattern we expect for a given branch."""
    if branch_name == "main":
        # Main should have stable versions (no dev, alpha, beta, rc suffixes)
        # If we have current version, use it, otherwise we'll look for it
        return "stable"
    elif branch_name == "dev":
        # Dev should have .devN versions
        return "dev"
    else:
        return None


def get_deployed_version_for_ref(ref_name, deployed_versions):
    """Find the deployed version directory that corresponds to a git ref.
    This handles the mapping between git refs and their deployed version directories.
    """
    # For main branch, look for the latest stable version (without dev suffix)
    if ref_name == "main":
        # First check if 'main' directory exists
        if "main" in deployed_versions:
            return "main"
        # Otherwise find the latest non-dev version
        stable_versions = [v for v in deployed_versions if not ("dev" in v or "alpha" in v or "beta" in v or "rc" in v)]
        if stable_versions:
            # Sort and get the latest
            return sorted(stable_versions, key=lambda x: normalize_version_for_comparison(x))[-1]

    # For dev branch, look for dev versions
    if ref_name == "dev":
        # First check if 'dev' directory exists
        if "dev" in deployed_versions:
            return "dev"
        # Otherwise find the latest dev version
        dev_versions = [v for v in deployed_versions if ".dev" in v]
        if dev_versions:
            # Sort and get the latest
            return sorted(dev_versions)[-1]

    # For tags, normalize and look for exact match
    normalized = normalize_version_for_comparison(ref_name)
    if normalized in deployed_versions:
        return normalized

    # Also check with 'v' prefix removed
    if ref_name.startswith("v") and ref_name[1:] in deployed_versions:
        return ref_name[1:]

    # Check if the ref itself is in deployed versions
    if ref_name in deployed_versions:
        return ref_name

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

    # Get deployed versions early to use for version detection
    deployed = get_deployed_versions()
    if not deployed and os.environ.get("GITHUB_ACTIONS") == "true":
        print("Warning: Could not get deployed versions. Tags will be added without checking deployment status.")

    branches_to_process = [
        {"id": "main", "display_name": "latest", "preferred": True},
        {"id": "dev", "display_name": "dev"},
    ]

    for branch_info in branches_to_process:
        branch_id = branch_info["id"]
        git_ref_for_remote_lookup = f"origin/{branch_id}"

        # Determine the version string to use
        if branch_id == current_ci_ref_name and local_dynamic_version:
            # We're currently building this branch, use the local version
            switcher_version_str = local_dynamic_version
            # For URLs, determine the deployed directory
            if branch_id == "main":
                # Main branch gets deployed to the version directory
                deployed_dir = normalize_version_for_comparison(local_dynamic_version)
            elif branch_id == "dev":
                # Dev branch might be deployed as version or as 'dev'
                if "dev" in deployed:
                    deployed_dir = "dev"
                else:
                    deployed_dir = local_dynamic_version
            else:
                deployed_dir = branch_id
            print(f"Using local dynamic version '{switcher_version_str}' for '{branch_id}' (currently building)")
        else:
            # Not currently building this branch - need to find its deployed version
            version_from_ref = get_version_from_git_ref(git_ref_for_remote_lookup)

            if version_from_ref:
                # Static version found in pyproject.toml
                switcher_version_str = version_from_ref
                deployed_dir = get_deployed_version_for_ref(branch_id, deployed) or branch_id
            else:
                # Dynamic versioning - find the actual deployed version
                pattern = get_expected_version_pattern(branch_id)

                if pattern == "stable":
                    # For main, find the latest stable version
                    stable_versions = [
                        v
                        for v in deployed
                        if not any(
                            suffix in v
                            for suffix in [
                                ".dev",
                                "alpha",
                                "beta",
                                "rc",
                                "-dev",
                                "-alpha",
                                "-beta",
                                "-rc",
                            ]
                        )
                    ]
                    if stable_versions:
                        # Sort versions properly (you might want to use packaging.version here)
                        import re

                        def version_key(v):
                            # Simple version sorting - extract numbers
                            parts = re.findall(r"\d+", v)
                            return [int(p) for p in parts] if parts else [0]

                        stable_versions.sort(key=version_key, reverse=True)
                        switcher_version_str = stable_versions[0]
                        deployed_dir = switcher_version_str
                    else:
                        # Fallback to 'main' if no stable version found
                        switcher_version_str = "main"
                        deployed_dir = "main"
                elif pattern == "dev":
                    # For dev, find the latest dev version
                    dev_versions = [v for v in deployed if ".dev" in v]
                    if dev_versions:
                        # Sort to get latest dev version
                        dev_versions.sort(reverse=True)
                        switcher_version_str = dev_versions[0]
                        deployed_dir = switcher_version_str
                    else:
                        # Fallback to 'dev' if no dev version found
                        switcher_version_str = "dev"
                        deployed_dir = "dev"
                else:
                    # Other branches
                    deployed_dir = get_deployed_version_for_ref(branch_id, deployed)
                    switcher_version_str = deployed_dir or branch_id

                print(f"Using version '{switcher_version_str}' for '{branch_id}' (dynamic versioning)")

        if remote_branch_exists(branch_id):
            entry = {
                "name": branch_info["display_name"],
                "version": switcher_version_str,
                "url": f"{base_url}/{deployed_dir}/",
            }
            if branch_info.get("preferred"):
                entry["preferred"] = True
            switcher.append(entry)
            processed_identifiers.add(branch_id)
        else:
            print(f"Remote branch 'origin/{branch_id}' not found. Skipping.")

    # Process tags
    all_tags = get_git_tags()
    print(f"Found git tags: {all_tags}")

    for tag in all_tags:
        if tag in processed_identifiers:
            continue

        # Find the deployed directory for this tag
        deployed_dir = get_deployed_version_for_ref(tag, deployed)

        if not deployed_dir:
            print(f"Tag '{tag}' not found in deployed versions. Skipping.")
            continue

        # Determine version string
        if tag == current_ci_ref_name and local_dynamic_version:
            # We're currently building this tag
            tag_version_str = local_dynamic_version
            print(f"CI is building tag '{tag}'. Using local dynamic version '{local_dynamic_version}'.")
        else:
            # Use the deployed directory name as the version
            # This ensures consistency with what's actually deployed
            tag_version_str = deployed_dir

        entry = {
            "name": normalize_version_for_comparison(tag),
            "version": tag_version_str,
            "url": f"{base_url}/{deployed_dir}/",
        }
        switcher.append(entry)
        processed_identifiers.add(tag)

    # Sort and finalize switcher
    final_switcher = []
    main_entry = next((e for e in switcher if e.get("preferred")), None)
    dev_entry = next((e for e in switcher if e["name"] == "dev" and not e.get("preferred")), None)

    if main_entry:
        final_switcher.append(main_entry)
    if dev_entry:
        final_switcher.append(dev_entry)

    other_entries = [e for e in switcher if e not in (main_entry, dev_entry)]
    # Sort by version string (you can use packaging.version for better sorting)
    other_entries.sort(key=lambda x: x["version"], reverse=True)
    final_switcher.extend(other_entries)

    # Write output
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, "switcher.json")

    with open(output_path, "w") as f:
        json.dump(final_switcher, f, indent=2)

    print(f"Switcher file generated at '{output_path}' with {len(final_switcher)} entries.")
    print("Switcher content:")
    print(json.dumps(final_switcher, indent=2))


if __name__ == "__main__":
    main()
