SHELL := bash
MKDIR_P := mkdir -p

RESULTSDIR := ../results
RUN := export PYTHONPATH=$$PWD/.. &&

.PHONY: help
help:
	cat Makefile

.PHONY: show-disable-gpu
show-disable-gpu:
	@echo 'export CUDA_VISIBLE_DEVICES=-1'


.PHONY: all
all: evaluate-synthetic evaluate-house evaluate-willow evaluate-pascal-voc evaluate-spair


.PHONY: iccv2021
iccv2021: evaluate-synthetic evaluate-house evaluate-willow evaluate-pascal-voc


.PHONY: evaluate-synthetic
evaluate-synthetic: evaluate-synthetic-noise evaluate-synthetic-outlier

.PHONY: evaluate-synthetic-noise
evaluate-synthetic-noise:
	$(RUN) python evaluate_synthetic_noise.py

.PHONY: evaluate-synthetic-outlier
evaluate-synthetic-outlier:
	$(RUN) python evaluate_synthetic_outlier.py


.PHONY: evaluate-house
evaluate-house: evaluate-house-gap evaluate-house-outlier

.PHONY: evaluate-house-gap
evaluate-house-gap:
	$(RUN) python evaluate_house_gap.py

.PHONY: evaluate-house-outlier
evaluate-house-outlier:
	$(RUN) python evaluate_house_outlier.py


.PHONY: evaluate-willow
evaluate-willow: evaluate-willow-out-0 evaluate-willow-out-5

.PHONY: evaluate-willow-out-0
evaluate-willow-out-0:
	$(RUN) export outlier=0 \
	  && export savedir=$(RESULTSDIR)/willow \
	  && $(MKDIR_P) "$${savedir}" \
	  && python -u evaluate_willow.py --outlier $${outlier} \
	       | tee "$${savedir}/evaluation_out_$${outlier}.txt"

.PHONY: evaluate-willow-out-5
evaluate-willow-out-5:
	$(RUN) export outlier=5 \
	  && export savedir=$(RESULTSDIR)/willow \
	  && $(MKDIR_P) "$${savedir}" \
	  && python -u evaluate_willow.py --outlier $${outlier} \
	       | tee "$${savedir}/evaluation_out_$${outlier}.txt"


.PHONY: evaluate-pascal-voc
evaluate-pascal-voc:
	export savedir=$(RESULTSDIR)/pascal_voc \
	  && $(MKDIR_P) "$${savedir}" \
	  && $(RUN) python -u evaluate_pascal_voc.py \
	       | tee "$${savedir}/evaluation.txt"


.PHONY: evaluate-spair
evaluate-spair:
	export savedir=$(RESULTSDIR)/spair \
	  && $(MKDIR_P) "$${savedir}" \
	  && $(RUN) python -u evaluate_spair.py \
	       | tee "$${savedir}/evaluation.txt"


.PHONY: redraw-from-pickles
redraw-from-pickles:
	$(RUN) python redraw_from_pickles.py


.PHONY: demo-house
demo-house:
	$(RUN) python demo_house.py

.PHONY: demo-pascal-voc
demo-pascal-voc:
	$(RUN) python demo_pascal_voc.py

.PHONY: demo-willow
demo-willow:
	$(RUN) python demo_willow.py


.PHONY: clean
clean:
	-find . -name '*.pyc' -exec rm -f {} +
	-find . -name '*.pyo' -exec rm -f {} +
	-find . -name '*~' -exec rm -f {} +
	-find . -name '__pycache__' -exec rm -fr {} +
	-find . -name '.pytest_cache' -exec rm -fr {} +
