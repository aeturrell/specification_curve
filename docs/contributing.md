# Contributing

Thank you for your interest in improving this project. This project is
open-source under the [MIT license](https://opensource.org/licenses/MIT)
and welcomes contributions in the form of bug reports, feature requests,
and pull requests.

Here is a list of important resources for contributors:

- [Source Code](https://github.com/aeturrell/specification_curve)
- [Documentation](https://specification_curve.readthedocs.io/)
- [Issue Tracker](https://github.com/aeturrell/specification_curve/issues)

## How to report a bug

Report bugs on the [Issue Tracker](https://github.com/aeturrell/specification_curve/issues).

When filing an issue, make sure to answer these questions:

-   Which operating system and Python version are you using?
-   Which version of this project are you using?
-   What did you do?
-   What did you expect to see?
-   What did you see instead?

The best way to get your bug fixed is to provide a test case, and/or
steps to reproduce the issue.

## How to request a feature

Request features on the [Issue
Tracker](https://github.com/aeturrell/specification_curve/issues).

## How to set up your development environment

You need Python 3.8+ and the following tools:

-   [Poetry](https://python-poetry.org/)
-   [Nox](https://nox.thea.codes/)
-   [nox-poetry](https://nox-poetry.readthedocs.io/)
-   [Make](https://www.gnu.org/software/make/)


Install the package with development requirements:

```bash
$ poetry install
```

You can now run an interactive Python session, or the command-line
interface:

```bash
$ poetry run python
$ poetry run specification_curve
```

## How to test the project

Run the full test suite:

```bash
$ nox
```

List the available Nox sessions:

```bash
$ nox --list-sessions
```

You can also run a specific Nox session. For example, invoke the unit
test suite like this:

```bash
$ nox --session=tests
```

Unit tests are located in the `tests` directory, and are written using
the [pytest](https://pytest.readthedocs.io/) testing framework.

## How to submit changes

Open a [pull request](https://github.com/aeturrell/specification_curve/pulls) to
submit changes to this project.

Your pull request needs to meet the following guidelines for acceptance:

-   The Nox test suite must pass without errors and warnings.
-   Include unit tests. This project maintains 100% code coverage.
-   If your changes add functionality, update the documentation
    accordingly.

Feel free to submit early, though---we can always iterate on this.

To run linting and code formatting checks before committing your change,
you can install pre-commit as a Git hook by running the following
command:

```bash
$ nox --session=pre-commit -- install
```

It is recommended to open an issue before starting work on anything.
This will allow a chance to talk it over with the owners and validate
your approach.

## How to build the documentation

The readme for this project is automatically generated.

- Run `make clean`
- Run `make`

To upload the documentation, it’s

```bash
poetry run ghp-import -n -p -f docs/_build/html
```
