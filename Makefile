SHELL := bash
MKDIR_P := mkdir -p

RESULTSDIR := results

.PHONY: all
all: compile

.PHONY: help
help:
	cat Makefile

.PHONY: compile
compile:
	make -C utils/compute_feature
	make -C methods/TM
	make -C methods/RRWHM
	make -C methods/BCAGM
	make -C methods/ADGM

.PHONY: test
test:
	make -C utils/compute_feature test
	cd tests && PYTHONPATH=$$PWD/.. pytest .
	make -C methods/TM test
	make -C methods/RRWHM test
	make -C methods/BCAGM test
	make -C methods/ADGM test
	@# Run on GPU
	make -C methods/HGNN test

.PHONY: data
data:
	make -C data all

.PHONY: show-house
show-house:
	python utils/house.py

.PHONY: show-pascal-voc
show-pascal-voc:
	python utils/pascal_voc.py

.PHONY: show-willow
show-willow:
	python utils/willow.py

.PHONY: compare-compute-feature-time
compare-compute-feature-time:
	cd tests && PYTHONPATH=$$PWD/.. python compare_compute_feature_time.py

.PHONY: clean-results
clean-results:
	-rm -rf $(RESULTSDIR)

.PHONY: clean-cache
clean-cache:
	-find . -name '*.pyc' -exec rm -f {} +
	-find . -name '*.pyo' -exec rm -f {} +
	-find . -name '*~' -exec rm -f {} +
	-find . -name '__pycache__' -exec rm -fr {} +
	-find . -name '.pytest_cache' -exec rm -fr {} +

.PHONY: clean
clean: clean-cache
	make -C utils/compute_feature clean
	make -C methods/TM clean
	make -C methods/RRWHM clean
	make -C methods/BCAGM clean
	make -C methods/ADGM clean
	make -C methods/HGNN clean

.PHONY: clean-all
clean-all: clean clean-results
	make -C methods/ADGM clean-all
	make -C methods/HGNN clean-all
	make -C data clean-all
