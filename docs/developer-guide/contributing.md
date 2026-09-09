# Contributing Guide

Thanks for taking the time to contribute! We appreciate all contributions, from reporting bugs to implementing new features. If it's unclear on how to proceed after reading this guide, you can ask on [Discord](https://discord.gg/dNfGMUyPa8).

## Opening an issue

You can report any issue by opening a [new issue](https://github.com/sovai-research/panelary/issues/new/choose).

**Bug reports** should include:

1. Your **OS, the Python version and Panelary (`panelary`) version** you are using.
2. A **minimal reproducible example (MRE)**, i.e. the code and some (fake) data that can be used to reproduce the error you encounter. It might take a bit more time on your side, but it greatly helps maintainers to solve your issue quickly.

**Feature requests** should also start from a dedicated issue, even if you plan to contribute to the feature yourself. In this way, maintainers can help you plan the design of the new feature and ease the development.

## Contributing to the codebase

Contributions should always start from an issue: even if you wish to contribute to Panelary's features, it is best to open a new issue so that the maintainers can help you through the design process.

### Picking an issue

Pick an issue by going through the [issue tracker](https://github.com/sovai-research/panelary/issues) and finding an issue you would like to work on. To work on an issue, please leave a new message below the discussion to show your interest. We use the [`help wanted`](https://github.com/sovai-research/panelary/labels/help%20wanted) label to indicate issues that are high on our wishlist. However, if you are a first time contributor, you might want to look for issues labeled [`good first issue`](https://github.com/sovai-research/panelary/labels/good%20first%20issue).

### Set up your local environment

Panelary is **pure Python** as of 0.4.0 -- the Rust extension is gone and the distribution is a
single universal `py3-none-any` wheel. There is no Rust toolchain to install and no compiler
step: a checkout plus a Python 3.10+ interpreter is the whole prerequisite list.

We use [**uv**](https://docs.astral.sh/uv/) for environments and installs. It is the same
installer CI uses (`.github/workflows/ci.yml`), and it resolves and installs an order of
magnitude faster than pip. Everything below also works with plain `pip` if you prefer -- see the
fallback at the end.

1. **Install uv** (one time):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh     # macOS / Linux
# Windows (PowerShell):
# powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

2. **Fork and clone the repository**:

```bash
# via gh CLI
gh repo fork sovai-research/panelary --clone

# or via ssh
git clone git@github.com:<your-username>/panelary.git
cd panelary
```

3. **Create the environment and install the project**. uv will download a managed CPython for
   you if the version you ask for is not already present:

```bash
uv python install 3.10        # optional: matches the minimum supported version
uv venv                        # creates ./.venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# editable install + dev tooling + the batteries-included optional deps
uv pip install -e ".[dev,recommended]"
```

4. **Install the pre-commit hooks**:

```bash
uv run pre-commit install --install-hooks
```

5. **Run the tests**:

```bash
# the 40-minute forecasting suite runs nightly, not on every change
uv run pytest -q --ignore=tests/test_forecasting.py
```

**Without uv.** Nothing here requires it; the equivalents are
`python3 -m venv .venv`, `python3 -m pip install -e ".[dev,recommended]"`,
`pre-commit install --install-hooks` and `pytest -q`. The [Makefile](https://github.com/sovai-research/panelary/blob/main/Makefile)
detects uv automatically and falls back to pip when it is absent, so
`make venv && make edit`, `make test`, `make lint` and `make typecheck` work either way.

### While working on your issue

Create a new git branch from the `main` branch in your local repository, and start coding!

The Python package lives under `panelary/` and the suite under `tests/`. To run the
tests:

```bash
uv run pytest -q --ignore=tests/test_forecasting.py
# or: make test
```

`pre-commit` checks will run before any commit. To lint, format and type-check the tree
yourself:

```bash
uv run ruff check --fix .
uv run ruff format .
# or: make fmt

uv run mypy panelary
# or: make typecheck
```

`mypy` is configured with `files = ["panelary"]` in `pyproject.toml`, so a bare `uv run mypy`
checks the same paths. Type-checking is **blocking in CI** (ratcheted against a baseline error
count), as is `ruff`. `make check` runs every gate — lint, type-check and tests — in one go.

Note that your work cannot be merged if these checks fail!

Two other things to keep in mind:

* Add test to your code. If you haven't written tests before, the dev team will be glad to help you out. We will link some useful resources here too.
* If you change the public API, update the documentation.

### Pull requests

When you have resolved your issue, [open a pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork) in the repository. Please adhere to the following guidelines:

* **Start your pull request title with a [conventional commit tag](https://www.conventionalcommits.org/en/v1.0.0/)**. This helps us add your contribution to the right section of the changelog. We use the [Angular](https://github.com/angular/angular/blob/22b96b9/CONTRIBUTING.md#type) convention.
* Use a descriptive title starting with an uppercase letter. This text will end up in the changelog.
* In the pull request description, [link](https://docs.github.com/en/issues/tracking-your-work-with-issues/linking-a-pull-request-to-an-issue) to the issue you were working on.
* Add any relevant information to the description that you think may help the maintainers review your code.
* Make sure your branch is [rebased](https://docs.github.com/en/get-started/using-git/about-git-rebase) against the latest version of the main branch.
* Make sure all GitHub Actions checks pass.
* After you have opened your pull request, a maintainer will review it and possibly leave some comments. Once all issues are resolved, the maintainer will merge your pull request, and your work will be part of the next Panelary release!

Keep in mind that your work does not have to be perfect right away! If you are stuck or unsure about your solution, feel free to open a draft pull request and ask for help.

## Contributing to the documentation

The site is [MkDocs](https://www.mkdocs.org/) with the Material theme; `mkdocs.yml` at the repo
root holds the nav and plugin configuration. Install the toolchain and serve it locally:

```bash
uv pip install -e ".[docs]"
uv run mkdocs serve          # live-reloading preview on http://127.0.0.1:8000
uv run mkdocs build --strict # what CI runs: any warning is an error
```

Three conventions keep the docs consistent:

* **`docs/api-reference/` pages are thin.** Each is a short hand-written introduction followed
  by a `## API` section containing a single `::: panelary.<module>` block; the reference itself
  is generated from the docstrings by [mkdocstrings](https://mkdocstrings.github.io/). Every
  page follows the same shape — `# Title`, lead paragraph, `## What's here`, any narrative
  sections, `## See also`, `## API`. Prose about *how* to use a module belongs in
  `docs/user-guide/` instead, linked from `## See also`.
* **Docstrings are NumPy style** (`docstring_style: numpy` in `mkdocs.yml`). Because the
  reference is generated, improving a docstring improves the site — no page edit needed.
* **A new page must be added to the `nav:` block** in `mkdocs.yml`, or it is built but
  unreachable.

`mkdocs build --strict` fails on a broken internal link or an unresolvable mkdocstrings
identifier, so run it before opening a documentation pull request.

## Credits

This guide is inspired from [Polars User guide](https://docs.pola.rs/development/contributing/).
