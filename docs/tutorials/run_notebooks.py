#!/usr/bin/env python3
"""
docs/tutorials/run_notebooks.py

For each tutorial dir (with pixi.toml + .ipynb):
  1) pixi config set --local run-post-link-scripts insecure
  2) pixi install
  3) pixi run python -m pip install --upgrade ipykernel papermill nbclient
  4) pixi run python -m papermill <notebook> build/<notebook>
"""
import subprocess
import sys
from pathlib import Path

def run(cmd, cwd):
    print(f"> {' '.join(cmd)}  (cwd={cwd.name})")
    subprocess.run(cmd, cwd=str(cwd), check=True)

def main():
    tutorials = Path(__file__).parent.resolve()
    for proj in sorted(tutorials.iterdir()):
        if not (proj / "pixi.toml").is_file():
            continue
        notebooks = list(proj.glob("*.ipynb"))
        if not notebooks:
            continue

        print(f"\n=== {proj.name} ===")

        # 1) Allow post-link scripts (so graphviz plugins are registered)
        run([
            "pixi", "config", "set", "--local",
            "run-post-link-scripts", "insecure"
        ], cwd=proj)

        # 2) Build or update the env from pixi.toml
        run(["pixi", "install"], cwd=proj)

        # 3) Bootstrap kernel & notebook tools
        run([
            "pixi", "run", "python", "-m", "pip", "install", "--upgrade",
            "ipykernel", "papermill", "nbclient"
        ], cwd=proj)

        # 4) Execute each notebook via papermill
        build = proj / "build"
        build.mkdir(exist_ok=True)
        for nb in notebooks:
            out = build / nb.name
            run([
                "pixi", "run", "python", "-m", "papermill",
                str(nb), str(out)
            ], cwd=proj)

    print("\n✅ All tutorials processed.")

if __name__ == "__main__":
    main()
