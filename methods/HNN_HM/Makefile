SHELL := bash

.PHONY: all
all:

.PHONY: test
test:
	export PYTHONPATH=$$PWD/..:$$PWD/../.. && cd tests/ && pytest .

.PHONY: debug
debug:
	export PYTHONPATH=$$PWD/../.. && python debug.py

.PHONY: clean
clean:
	-find . -name '*.pyc' -exec rm -f {} +
	-find . -name '*.pyo' -exec rm -f {} +
	-find . -name '*~' -exec rm -f {} +
	-find . -name '__pycache__' -exec rm -fr {} +
	-find . -name '.pytest_cache' -exec rm -fr {} +

.PHONY: clean-results
clean-results:
	-rm -rf "results"

.PHONY: clean-all
clean-all: clean clean-results
