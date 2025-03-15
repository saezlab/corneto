import json
import os
import subprocess
import sys


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
    """Fetches the gh-pages branch and returns a set of directory names that have been deployed.
    These directories correspond to the versions (branch names or tags) for which pages exist.
    """
    try:
        # Fetch the gh-pages branch into a local branch named "gh-pages"
        subprocess.check_call(
            ["git", "fetch", "origin", "gh-pages:gh-pages"], text=True
        )
        # List directories at the root of gh-pages
        output = subprocess.check_output(
            ["git", "ls-tree", "-d", "--name-only", "gh-pages"], text=True
        )
        deployed = set(line.strip() for line in output.splitlines() if line.strip())
        return deployed
    except subprocess.CalledProcessError:
        return set()


def get_static_folder():
    """Import the Sphinx conf.py to get the html_static_path setting.
    Returns the absolute path to the first static folder defined.
    """
    current_dir = os.path.dirname(__file__)
    sys.path.insert(0, current_dir)
    try:
        import conf  # conf.py should be in the same folder (docs)

        static_rel = conf.html_static_path[0] if conf.html_static_path else "_static"
    except Exception as e:
        print("Could not import conf.py, falling back to '_static'.", e)
        static_rel = "_static"
    return os.path.join(current_dir, static_rel)


def main():
    # Extract GitHub username from GITHUB_REPOSITORY (format: username/repository)
    repo = os.environ.get("GITHUB_REPOSITORY", "username/corneto")
    username = repo.split("/")[0]

    # Build the base URL using the GitHub username.
    base_url = f"https://{username}.github.io/corneto"

    switcher = []

    # Process branches: include them (with "main" always preferred).
    for branch in ["main", "dev"]:
        if remote_branch_exists(f"origin/{branch}"):
            entry = {"name": branch, "version": branch, "url": f"{base_url}/{branch}/"}
            if branch == "main":
                entry["preferred"] = True
            switcher.append(entry)

    # Get the deployed versions from the gh-pages branch.
    deployed = get_deployed_versions()
    if not deployed:
        print(
            "Warning: Could not determine deployed versions; no filtering of tags will be applied."
        )

    # Process tags: include only tags for which a directory exists on gh-pages.
    for tag in get_git_tags():
        # If we have deployed versions, only include tag if it exists there.
        if deployed and tag not in deployed:
            print(f"Skipping tag '{tag}': no deployed pages found.")
            continue
        entry = {"name": tag, "version": tag, "url": f"{base_url}/{tag}/"}
        switcher.append(entry)

    # Write the switcher to the JSON file in the static folder (as defined in conf.py).
    static_folder = get_static_folder()
    os.makedirs(static_folder, exist_ok=True)
    output_path = os.path.join(static_folder, "switcher.json")
    with open(output_path, "w") as f:
        json.dump(switcher, f, indent=2)

    print(f"Switcher file generated at {output_path}")


if __name__ == "__main__":
    main()
