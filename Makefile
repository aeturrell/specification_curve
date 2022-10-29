# This makes the documentation and readme for specification_curve

.PHONY: all clean site

all: README.md site

# Build the readme
README.md: docs/index.ipynb
		poetry run jupyter nbconvert --to markdown --execute docs/index.ipynb \
		&& mv docs/index.md README.md \
		&& poetry run nbstripout docs/index.ipynb



# Build the github pages site
site:
		poetry run jupyter-book build docs/

clean:
	rm README.md
