EIGEN_VERSION := 3.3.1
EIGEN := eigen-$(EIGEN_VERSION)

INC := $(shell python -m pybind11 --includes) -I core/include -I $(EIGEN)

SRC := core/src/ADGM.cpp pybind11/wrapADGM.cpp
DST := ADGMCore.so

.PHONY: all
all: $(DST) 

$(DST): $(EIGEN) $(SRC)
	c++ -O3 -Wall -shared -std=c++11 -fPIC $(INC) $(SRC) -o $@

$(EIGEN):
	curl -sSL "https://gitlab.com/libeigen/eigen/-/archive/$(EIGEN_VERSION)/$(EIGEN).tar.gz" | tar xz

.PHONY: test
test: $(DST)
	PYTHONPATH="$$PWD/.." pytest tests/

.PHONY: clean
clean:
	-find . -name '*.pyc' -exec rm -f {} +
	-find . -name '*.pyo' -exec rm -f {} +
	-find . -name '*~' -exec rm -f {} +
	-find . -name '__pycache__' -exec rm -fr {} +
	-find . -name '.pytest_cache' -exec rm -fr {} +
	-rm -f $(DST)

.PHONY: clean-all
clean-all: clean
	-rm -rf "$(EIGEN)"
