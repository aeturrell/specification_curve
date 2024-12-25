# This makes the documentation for specification_curve
# In practice, though, done in the GitHub Action (release)
.PHONY: all site publish

all: site

# Build the github pages site
site:
		uv pip install -e .
		uv run quartodoc build --config docs/_quarto.yml
		cd docs; uv run quarto render --execute
		rm docs/.gitignore
		uv run nbstripout docs/*.ipynb
		uv run pre-commit run --all-files


publish:
		uv pip install -e .
		uv run quartodoc build --config docs/_quarto.yml
		cd docs;uv run quarto render --execute
		cd docs;uv run quarto publish gh-pages --no-render
		rm docs/.gitignore
		uv run nbstripout docs/*.ipynb
		uv run pre-commit run --all-files
