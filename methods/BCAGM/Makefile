INC := $(shell python -m pybind11 --includes) -I core/include -I pybind11/include

SRC := $(wildcard core/src/*.cpp pybind11/src/*.cpp)
DST := BCAGMCore.so

.PHONY: all
all: $(DST) 

$(DST): $(SRC)
	c++ -O3 -Wall -shared -std=c++11 -fPIC $(INC) $^ -o $@

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
