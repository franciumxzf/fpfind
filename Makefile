PACKAGE=pfind
DIR=src/${PACKAGE}
LIB=${DIR}/lib

#.SILENT:
.PHONY: clean

all:
	gcc -o ${DIR}/freqcd ${DIR}/freqcd.c ${LIB}/getopt.c
	gcc -o tests/sample/example_hello tests/sample/example_hello.c

# Package-related stuff
test:
	poetry run pytest
clean:
	find . -type d -name ".pytest_cache" | xargs rm -rf
	rm -f ${DIR}/freqcd
	rm -f tests/sample/example_hello

# Python package management
install: install-poetry
	poetry install
install-poetry:
	pip install -U poetry
uninstall:
	poetry run pip uninstall ${PACKAGE}


_all:
	./generate_freqcd_testcase.py -t 0 1000000000 2000000000 |\
	   	./freqcd -f 1000000 |\
		./freqcd -f -999933 -o .output

