# Contributing

Thank you for your interest in improving this project. This project is
open-source under the [MIT license](https://opensource.org/licenses/MIT)
and welcomes contributions in the form of bug reports, feature requests,
and pull requests.

Here is a list of important resources for contributors:

- [Source Code](https://github.com/aeturrell/specification_curve)
- [Documentation](https://aeturrell.github.io/specification_curve/)
- [Issue Tracker](https://github.com/aeturrell/specification_curve/issues)

## How to report a bug

Report bugs on the [Issue Tracker](https://github.com/aeturrell/specification_curve/issues).

When filing an issue, make sure to answer these questions:

- Which operating system and Python version are you using?
- Which version of this project are you using?
- What did you do?
- What did you expect to see?
- What did you see instead?

The best way to get your bug fixed is to provide a test case, and/or
steps to reproduce the issue.

## How to request a feature

Request features on the [Issue
Tracker](https://github.com/aeturrell/specification_curve/issues).

## How to set up your development environment

You need Python 3.9+ and the following tools:

- [Poetry](https://python-poetry.org/)
- [Nox](https://nox.thea.codes/)
- [nox-poetry](https://nox-poetry.readthedocs.io/)
- [Make](https://www.gnu.org/software/make/) (for documentation)
- [Quarto](https://quarto.org/) (for documentation)

Before you install the environment using poetry, you may wish to run `poetry config virtualenvs.in-project true
` to get the virtual environment in the same folder as the code.

Install the package with development requirements:

```bash
poetry install
```

You can now run an interactive Python session, or the command-line interface.

## How to test the project

Run the full test suite:

```bash
nox
```

List the available Nox sessions:

```bash
nox --list-sessions
```

You can also run a specific Nox session. For example, invoke the unit
test suite like this:

```bash
nox --session=tests
```

Unit tests are located in the `tests` directory, and are written using
the [pytest](https://pytest.readthedocs.io/) testing framework.

## How to submit changes

Open a [pull request](https://github.com/aeturrell/specification_curve/pulls) to
submit changes to this project.

Your pull request needs to meet the following guidelines for acceptance:

- The Nox test suite must pass without errors and warnings.
- Include unit tests. This project maintains 100% code coverage.
- If your changes add functionality, update the documentation accordingly.

Feel free to submit early, though---we can always iterate on this.

Linting and formatting are run as part of pre-commit and nox. To just run pre-commit checks, use `poetry run pre-commit run --all-files`.

We recommend that you open an issue before starting work on any new features. This will allow a chance to talk it over with the owners and validate your approach.

## How to build the documentation

You can build the docs locally to look at it. The command is `make`: this will build the docs and put them in `docs/_site/`.

To publish new docs to GitHub Pages (where the documentation is displayed as web pages), it’s `make publish`—but only devs with admin rights will be able to execute this.

## How to create a package release

- Open a new branch with the version name

- Change the version in pyproject.toml

- Commit the change with a new version label as the commit message (checking the tests pass)

- Head to github and merge into main

- Draft a new release based on that most recent merge commit, using the new version as the tag

- Confirm the release draft on gitub

- The automatic release github action will push to PyPI.

If you ever need distributable files, you can use the `poetry build` command locally.
