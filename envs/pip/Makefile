SHELL := bash
MKDIR_P := mkdir -p

# cuda-10.0 + cudnn-7.6.2
CUDA_HOME := /usr/local/cuda

# Note(Xiaowei): The "&&" at the end is necessary,
RUN_IN_ENV := export CUDA_HOME="$(CUDA_HOME)" \
    && export LD_LIBRARY_PATH="$(CUDA_HOME)/extras/CUPTI/lib64:$(CUDA_HOME)/lib64/$${LD_LIBRARY_PATH:+:$$LD_LIBRARY_PATH}" \
    &&

.PHONY: help
help:
	cat Makefile

.PHONY: env
env:
	$(RUN_IN_ENV) pip install "pip>=19" && pip install -q -i https://mirrors.aliyun.com/pypi/simple -r ../../requirements.txt

.PHONY: run
run:
	$(RUN_IN_ENV) cd ../.. && bash --noprofile --norc -i

.PHONY: show
show:
	@echo '$(RUN_IN_ENV) true'

.PHONY: clean
clean:
	conda env remove -y -n $(CONDA_ENV)
