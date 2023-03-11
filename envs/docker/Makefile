image := liaoxw/hgm

# set environment variable `HOME` to avoid error when running `python -m pybind11 --includes`
RUN_IN_ENV := docker run -P --rm -it \
	    --gpus='all' \
	    -e 'HOME=/tmp' \
	    -u "$$(id -u):$$(id -g)" \
	    -v "$$(pwd)/../..:/workspace" \
	    -w /workspace \
	    $(image) 

.PHONY: help
help:
	cat Makefile

.PHONY: env
env:
	docker build -t $(image) .

.PHONY: run
run:
	$(RUN_IN_ENV) /bin/bash

.PHONY: show
show:
	@echo '$(RUN_IN_ENV)'

.PHONY: clean
clean:
	docker image rm $(image)
