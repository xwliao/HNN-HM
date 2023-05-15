SHELL := bash
MKDIR_P := mkdir -p

CURDIR := $(shell pwd)

DATA := $(CURDIR)
ADGM := $(CURDIR)/../methods/ADGM

SAVEDIR := $(CURDIR)/files
URL_FILE := $(CURDIR)/urls.txt
MD5SUM_FILE := $(CURDIR)/md5sum.txt

.PHONY: help
help:
	@echo "Usage:"
	@echo "  make all"
	@echo "  make download"
	@echo "  make check"
	@echo "  make extract"

.PHONY: all
all:
	@# Download files if the checking fail
	make check || make download
	@# Extract files if the checking succeed
	make check && make extract

.PHONY: download
download:
	@echo 'Note: pip install gdown (https://github.com/wkentaro/gdown)'
	$(MKDIR_P) "$(SAVEDIR)" && cd "$(SAVEDIR)" && \
	for url in $$(cat "$(URL_FILE)"); do \
	    gdown $$url; \
	done

.PHONY: check
check:
	md5sum -c < "$(MD5SUM_FILE)"

.PHONY: extract
extract:
	$(MKDIR_P) "$(DATA)"
	@# Eigen
	tar xzf "$(SAVEDIR)/eigen-3.3.1.tar.gz" -C "$(ADGM)/"
	@# CMU House Dataset
	tar xzf "$(SAVEDIR)/hotel.tar.gz" -C "$(DATA)/"
	@# CMU Hotel Dataset
	tar xzf "$(SAVEDIR)/house.tar.gz" -C "$(DATA)/"
	@# Willow Object Dataset
	unzip "$(SAVEDIR)/WILLOW-ObjectClass_dataset.zip" -d "$(DATA)/WILLOW-ObjectClass_dataset/"
	@# PASCAL VOC Dataset
	mkdir -p "$(DATA)/PascalVOC" \
	 && tar xzf "$(SAVEDIR)/voc2011_keypoints_Feb2012.tgz" -C "$(DATA)/PascalVOC/" \
	 && tar xf "$(SAVEDIR)/VOCtrainval_25-May-2011.tar" -C "$(DATA)/PascalVOC/"
	@# Cars and Motorbikes Graph Matching Dataset
	unzip "$(SAVEDIR)/Cars_and_Motorbikes_Dataset_and_Code.zip" -d "$(DATA)/"
	@# SPair-71k Dataset
	tar xzf "$(SAVEDIR)/SPair-71k.tar.gz" -C "$(DATA)/"

.PHONY: clean-extracted-files
clean-extracted-files:
	-rm -rf "$(ADGM)/eigen-3.3.1"
	-rm -rf "$(DATA)/house"
	-rm -rf "$(DATA)/hotel"
	-rm -rf "$(DATA)/WILLOW-ObjectClass_dataset"
	-rm -rf "$(DATA)/PascalVOC/annotations"
	-rm -rf "$(DATA)/PascalVOC/TrainVal"
	-rm -rf "$(DATA)/Cars_and_Motorbikes_Graph_Matching_Datasets_and_Code"

.PHONY: clean-downloaded-files
clean-downloaded-files:
	-rm -rf "$(SAVEDIR)"

.PHONY: clean
clean: clean-extracted-files

.PHONY: clean-all
clean-all: clean-extracted-files clean-downloaded-files
