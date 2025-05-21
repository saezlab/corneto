import json
import os
import re
import subprocess


def get_git_tags():
    """Return a list of git tags."""
    try:
        output = subprocess.check_output(["git", "tag"], text=True)
        return [line.strip() for line in output.splitlines() if line.strip()]
    except subprocess.CalledProcessError:
        return []


def remote_branch_exists(branch):
    """Return True if the given remote branch exists (e.g. 'origin/main')."""
    try:
        output = subprocess.check_output(["git", "branch", "-r"], text=True)
        return any(branch in line for line in output.splitlines())
    except subprocess.CalledProcessError:
        return False


def get_deployed_versions():
    """Fetch the gh-pages branch and return a set of directory names
    that have been deployed (e.g. "main", "dev", or tag names).
    """
    try:
        subprocess.check_call(
            ["git", "fetch", "origin", "gh-pages:gh-pages"], text=True
        )
        output = subprocess.check_output(
            ["git", "ls-tree", "-d", "--name-only", "gh-pages"], text=True
        )
        return {line.strip() for line in output.splitlines() if line.strip()}
    except subprocess.CalledProcessError:
        return set()


def get_version_from_branch(branch):
    """Extract the package version from pyproject.toml on the given branch.

    This function attempts to read the pyproject.toml file from the branch
    and extract the version value using a regular expression.
    """
    try:
        # Adjust the path below if your pyproject.toml is located elsewhere.
        file_path = "pyproject.toml"
        output = subprocess.check_output(
            ["git", "show", f"origin/{branch}:{file_path}"], text=True
        )
        match = re.search(r'^version\s*=\s*["\']([^"\']+)["\']', output, re.M)
        if match:
            return match.group(1)
    except subprocess.CalledProcessError:
        pass
    # Fallback: if extraction fails, return the branch name
    return branch


def get_base_url():
    # 1) try the real published URL
    url = os.environ.get("PAGE_URL")
    if url:
        print(f"Using PAGE_URL: {url}")
        return url.rstrip("/")
    # 2) fallback to github.io
    repo = os.environ.get("GITHUB_REPOSITORY", "username/corneto")
    user, project = repo.split("/", 1)
    print(f"Using {user}.github.io/{project} as fallback for the switcher URL.")
    return f"https://{user}.github.io/{project}"


def main():
    # Derive GitHub username from the environment variable.
    #repo = os.environ.get("GITHUB_REPOSITORY", "username/corneto")
    #username = repo.split("/")[0]
    #base_url = f"https://{username}.github.io/corneto"
    base_url = get_base_url()
    root_path = os.environ.get("SITE_ROOT", "").rstrip("/")

    switcher = []

    # Process branches: include "main" (preferred/stable) and "dev".
    for branch in ["main", "dev"]:
        if remote_branch_exists(f"origin/{branch}"):
            # Instead of using the branch name, get the real package version from pyproject.toml.
            version = get_version_from_branch(branch)
            entry = {
                "name": "latest" if branch == "main" else branch,
                "version": version,
                "url": f"{base_url}/{branch}/",
            }
            if branch == "main":
                entry["preferred"] = True
            switcher.append(entry)

    # Filter tags: include only tags that have a deployed directory.
    deployed = get_deployed_versions()
    if not deployed:
        print(
            "Warning: Could not determine deployed versions; no filtering of tags will be applied."
        )

    for tag in get_git_tags():
        if deployed and tag not in deployed:
            print(f"Skipping tag '{tag}' as it's not deployed.")
            continue
        entry = {"name": tag, "version": tag, "url": f"{base_url}/{tag}/"}
        switcher.append(entry)

    # Write the switcher file into docs/switcher.json.
    output_path = os.path.join(os.path.dirname(__file__), "switcher.json")
    with open(output_path, "w") as f:
        json.dump(switcher, f, indent=2)

    print(f"Switcher file generated at {output_path}")


if __name__ == "__main__":
    main()
