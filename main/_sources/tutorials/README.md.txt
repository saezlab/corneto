# 🧠 Contribute a Tutorial to CORNETO

We welcome contributions of tutorials that showcase how to use CORNETO for biological network inference, optimisation, and analysis. If you’ve built something useful or interesting with CORNETO, share it to help others learn!

All tutorials are stored in:
```
corneto/docs/tutorials/
```

---

## 📁 Directory Structure

Each tutorial must be placed in its own folder inside one of the following:

```
corneto/docs/tutorials/
  accepted/
    my_tutorial/
      notebook.ipynb
      requirements.txt  # or dependencies.yaml
  contrib/
    another_example/
      example.ipynb
      dependencies.yaml
  template/
    template.ipynb
    dependencies.yaml
```

- `accepted/`: Reviewed and validated tutorials.
- `contrib/`: Community submissions pending review.
- `template/`: A suggested structure to help you start.

---

## 📦 Dependency File Format

Each tutorial folder **must include** a file listing additional dependencies:
- Use either `requirements.txt` **or** `dependencies.yaml`
- **All packages must specify a minimum version (`>=`)**
- Avoid pinning (`==`) unless absolutely required

### ✅ Example `requirements.txt`
```text
scanpy>=1.9
matplotlib>=3.5
```

### ✅ Example `dependencies.yaml`
```yaml
dependencies:
  - scanpy>=1.9
  - matplotlib>=3.5
```

> ℹ️ If your tutorial has no extra dependencies, include an empty file or write:
> `# No additional dependencies required`

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
- `requirements.txt` or `dependencies.yaml` with **minimum versions**

### 6. Commit and Push

Use **[Conventional Commits](https://www.conventionalcommits.org/)**:

```bash
git add corneto/docs/tutorials/contrib/my_tutorial_name/
git commit -m "docs(tutorial): add my_tutorial_name example notebook"
git push origin contrib/my-tutorial-name
```

### 7. Open a Pull Request

Open a **PR against `dev`** with title:
```
docs(tutorial): add my_tutorial_name example
```

In the PR description, include:
- 📘 What the tutorial covers
- 🔧 Which CORNETO functionality it uses
- 🧪 External dependencies (with versions)
- 📊 Key outputs or takeaways

---

## 🧪 Tips for a Good Tutorial

- Use markdown cells to explain key steps and reasoning.
- Keep it concise and focused on a clear use-case.
- Use small or public datasets when possible.
- Ensure the notebook runs top to bottom without errors.
- Avoid heavy or obscure dependencies unless necessary.
- Use open source solvers such as `highs` or `scip` if possible.
- Include a README in your tutorial folder with:
  - A brief description of the tutorial
  - Instructions to run the notebook
  - Any additional notes or tips

---

## ✨ Recognition

Accepted tutorials will:
- Be moved to `accepted/`
- Be listed in CORNETO documentation
- Credit the author in the README or release notes

---

## 🧰 Questions?

Open a GitHub issue or start a discussion. We’re happy to help!

---

Happy contributing! 🧬
