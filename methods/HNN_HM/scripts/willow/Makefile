SHELL := bash

# Set this for each dataset
dataset_name := Willow
result_dir_name := willow

project_dir := $(shell pwd)/../../../..
work_dir := $(shell pwd)/../..

result_dir := $(work_dir)/results/$(result_dir_name)
checkpoint_dir := $(result_dir)/checkpoints
log_dir := $(result_dir)/logs

timestamp := $(shell date +%Y%m%d_%H%M%S)

.PHONY: all
all:

.PHONY: train
train:
	-rm -rf "$(checkpoint_dir)"
	mkdir -p "$(checkpoint_dir)"
	-rm -rf "$(log_dir)"
	mkdir -p "$(log_dir)"
	cd $(work_dir) && export PYTHONPATH=$(project_dir) \
	 && python train.py "$(dataset_name)" \
	      |& tee "$(log_dir)/logs_$(timestamp).txt"

.PHONY: debug
debug:
	cd $(work_dir) && export PYTHONPATH=$(project_dir) \
	 && python debug.py "$(dataset_name)"

.PHONY: tensorboard
tensorboard:
	tensorboard --logdir=$(log_dir)

.PHONY: clean
clean:
	-find . -name '*.pyc' -exec rm -f {} +
	-find . -name '*.pyo' -exec rm -f {} +
	-find . -name '*~' -exec rm -f {} +
	-find . -name '__pycache__' -exec rm -fr {} +
	-find . -name '.pytest_cache' -exec rm -fr {} +

.PHONY: clean-results
clean-results:
	-rm -rf "$(checkpoint_dir)"
	-rm -rf "$(log_dir)"
	-rm -rf "$(result_dir)"

.PHONY: clean-all
clean-all: clean clean-results
