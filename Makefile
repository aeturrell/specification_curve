# This makes the documentation for specification_curve
.PHONY: all site publish

all: site

# Build the github pages site
site:
		poetry run quartodoc build --config docs/_quarto.yml
		cd docs; poetry run quarto render --execute
		rm docs/.gitignore
		poetry run nbstripout docs/*.ipynb
		poetry run pre-commit run --all-files


publish:
		poetry run quartodoc build --config docs/_quarto.yml
		cd docs;poetry run quarto render --execute
		cd docs;poetry run quarto publish gh-pages --no-render
		rm docs/.gitignore
		poetry run nbstripout docs/*.ipynb
		poetry run pre-commit run --all-files
