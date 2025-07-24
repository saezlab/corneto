# 🧠 Contribute a Tutorial to CORNETO

We welcome contributions of tutorials that showcase how to use CORNETO for biological network inference, optimisation, and analysis. If you’ve built something useful or interesting with CORNETO, share it to help others learn!

All tutorials are stored in:
```
corneto/docs/tutorials/
```

---

## 📁 Directory Structure

Each tutorial is **independent** and must be placed in its own folder with its own `pixi.toml` file:

```
corneto/docs/tutorials/
  carnival/                    # Official tutorial
    notebook.ipynb
    pixi.toml
  fba/                        # Official tutorial
    notebook.ipynb
    pixi.toml
  contrib/                    # Community contributions
    my_tutorial/
      notebook.ipynb
      pixi.toml
    another_example/
      example.ipynb
      pixi.toml
```

- Official tutorials are maintained by the CORNETO team
- `contrib/`: Community submissions that go through PR review

---

## 📦 Environment Setup with Pixi

Each tutorial folder **must include** a `pixi.toml` file that:
- Specifies a **specific version** of corneto (not latest)
- Includes all additional dependencies needed for the tutorial
- Allows tutorials to be run independently of the main repository

### ✅ Example `pixi.toml`
```toml
[workspace]
name = "my_tutorial"
channels = ["conda-forge"]
platforms = ["linux-64", "osx-64", "osx-arm64", "win-64"]

[dependencies]
python = "3.10.*"
matplotlib = ">=3.5"
pip = "*"

[pypi-dependencies]
corneto = "==0.9.2"  # Use specific version, not latest
scanpy = ">=1.9"
# Add other dependencies as needed
```

> ℹ️ **Important**: Always specify a specific corneto version (e.g., `==0.9.2`) to ensure reproducibility and independence from the main repository.

---

## 🚀 How to Submit a Tutorial

### 1. Fork and Clone the Repository
```bash
git clone https://github.com/saezlab/corneto
cd corneto
```

### 2. Switch to the `dev` Branch
```bash
git checkout dev
```

### 3. Create a New Branch
```bash
git checkout -b contrib/my-tutorial-name
```

### 4. Set Up Pre-commit Hooks (Required)
We use pre-commit hooks to ensure code quality and consistency. Install and set up pre-commit before making any commits:

```bash
poetry install  # If you haven't installed dependencies yet
poetry run pre-commit install
```

This will automatically check your code with `ruff` when you commit changes.

### 5. Add Your Tutorial Folder
Place your folder here:
```
corneto/docs/tutorials/contrib/my_tutorial_name/
```
Include:
- `my_tutorial_name.ipynb`
- `pixi.toml` with specific corneto version and dependencies

### 6. Test Your Tutorial
Before submitting, test your tutorial independently:
```bash
cd corneto/docs/tutorials/contrib/my_tutorial_name/
pixi install
pixi run jupyter notebook
```

### 7. Commit and Push

Use **[Conventional Commits](https://www.conventionalcommits.org/)**:

```bash
git add corneto/docs/tutorials/contrib/my_tutorial_name/
git commit -m "docs(tutorial): add my_tutorial_name example notebook"
git push origin contrib/my-tutorial-name
```

### 8. Open a Pull Request

Open a **PR against `dev`** with title:
```
docs(tutorial): add my_tutorial_name example
```

In the PR description, include:
- 📘 What the tutorial covers
- 🔧 Which CORNETO functionality it uses
- 🧪 External dependencies (with versions)
- 📊 Key outputs or takeaways
- ✅ Confirm the tutorial runs independently with `pixi install && pixi run jupyter notebook`

---

## 🧪 Tips for a Good Tutorial

- Use markdown cells to explain key steps and reasoning.
- Keep it concise and focused on a clear use-case.
- Use small or public datasets when possible.
- Ensure the notebook runs top to bottom without errors.
- Avoid heavy or obscure dependencies unless necessary.
- Use open source solvers such as `highs` or `scip` if possible.
- **Pin a specific corneto version** in your `pixi.toml` for reproducibility.
- Test your tutorial in isolation using `pixi install && pixi run jupyter notebook`.
- Include a README in your tutorial folder with:
  - A brief description of the tutorial
  - Instructions to run the notebook with pixi
  - Any additional notes or tips

---

## ✨ Recognition

Merged tutorials will:
- Be listed in CORNETO documentation
- Credit the author in the README or release notes

---

## 🧰 Questions?

Open a GitHub issue or start a discussion. We’re happy to help!

---

Happy contributing! 🧬
