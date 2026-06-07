.PHONY: build install requirements requirements-mamba requirements-conda install-dev dev dev-mamba dev-conda docs docs-install docs-serve test lint autoformat clean coverage

requirements:
	python -m pip install -r requirements.txt

requirements-mamba:
	mamba install --file requirements.txt

requirements-conda:
	conda install --file requirements.txt

build:
	python -m build -o wheelhouse

install:
	pip install .

install-dev: 
	pip install --editable .

dev:
	python -m pip install --upgrade -r requirements-dev.txt

dev-mamba:
	mamba install --file requirements-dev.txt

dev-conda:
	conda install --file requirements-dev.txt

docs-install:
	python -m pip install -r requirements-docs.txt

# Build the static site to ./site. Uses griffe static analysis — does NOT
# import nsosim, so opensim/torch/NSM are not required to build the docs.
#
# Not --strict: strict escalates griffe's docstring-style warnings (missing
# param annotations, etc.) across the whole library into errors, which is out
# of scope to clean. The doc cross-reference staleness gate lives in
# tests/docs/test_doc_references.py instead.
docs:
	mkdocs build

docs-serve:
	mkdocs serve


test:
	set -e
	pytest

lint:
	set -e
	isort -c .
	black --check --config pyproject.toml .

autoformat:
	set -e
	isort .
	black --config pyproject.toml .

clean:
	rm -rf build dist *.egg-info  

# coverage: 
# 	coverage run -m pytest
# 	# Make the html version of the coverage results. 
# 	coverage html 