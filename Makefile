# This makes the documentation and readme for specification_curve

.PHONY: all clean site

all: site

# Build the github pages site
site:
	poetry run jupyter-book build docs/

clean:
	rm -rf docs/_build
