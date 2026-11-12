# This makes the documentation for specification_curve
# In practice, though, done in the GitHub Action (release)
.PHONY: all site preview publish

all: site

# Build the github pages site into great-docs/_site
site:
		uv pip install -e .
		uv run python scripts/make_readme_figure.py
		uv run great-docs build

# Serve the built site locally
preview:
		uv run great-docs preview

publish: site
		cd great-docs;uv run quarto publish gh-pages --no-render
