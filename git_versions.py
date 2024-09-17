import re
import subprocess

# Step 1: Get all commits where the version line in pyproject.toml changed using git log -G
try:
    # Escape the regex properly for subprocess
    log_command = [
        "git",
        "log",
        '-Gversion\\s*=\\s*"',
        "--format=%H",
        "--",
        "pyproject.toml",
    ]
    result = subprocess.run(log_command, capture_output=True, text=True, check=True)
    commit_hashes = result.stdout.strip().split("\n")
except subprocess.CalledProcessError as e:
    print(f"Error running git log command: {e}")
    print(f"stdout: {e.stdout}")
    print(f"stderr: {e.stderr}")
    commit_hashes = []


# Debugging: Print commit hashes to ensure the subprocess run works
print("Commit hashes retrieved:")
print(len(commit_hashes))

# Regular expression to find the [tool.poetry] section
poetry_section_regex = re.compile(r"^\[tool\.poetry\]", re.MULTILINE)
# Regular expression to capture the version line within the [tool.poetry] section
version_regex = re.compile(r'^version\s*=\s*"(.*?)"', re.MULTILINE)

# Step 2: Loop through each commit and extract the version from the [tool.poetry] section
for commit_hash in commit_hashes:
    if commit_hash:
        # Get the content of pyproject.toml at the specific commit
        try:
            show_command = ["git", "show", f"{commit_hash}:pyproject.toml"]
            toml_content = subprocess.run(
                show_command, capture_output=True, text=True, check=True
            ).stdout
        except subprocess.CalledProcessError as e:
            print(f"Error running git show command for commit {commit_hash}: {e}")
            print(f"stdout: {e.stdout}")
            print(f"stderr: {e.stderr}")
            continue

        # Check if [tool.poetry] section exists
        poetry_section_match = poetry_section_regex.search(toml_content)

        if poetry_section_match:
            # Start looking for the version line after the [tool.poetry] section
            start_pos = poetry_section_match.end()
            toml_content_from_poetry = toml_content[start_pos:]

            # Find the version line in this section
            version_match = version_regex.search(toml_content_from_poetry)

            if version_match:
                version = version_match.group(1)
                print(f"Version in [tool.poetry]: {version}, hash {commit_hash}")
            else:
                print("No version found in [tool.poetry] section.")
        else:
            print("No [tool.poetry] section found.")
