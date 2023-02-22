PACKAGE=pfind
DIR=src/${PACKAGE}
LIB=${DIR}/lib

# For 'read' and 'readx' rules, parse command line arguments
# from $(MAKECMDGOALS) and turn them into do-nothing targets.
# https://stackoverflow.com/questions/2214575/passing-arguments-to-make-run
ifneq (,$(findstring read,$(firstword $(MAKECMDGOALS))))
  READ_ARGS := $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))
  $(eval $(READ_ARGS):;@:)
endif

.SILENT: shell
.PHONY: clean,read,readx

all:
	gcc -o ${DIR}/freqcd ${DIR}/freqcd.c ${LIB}/getopt.c
shell:
	-poetry shell
pfind-shell:
	-poetry run python -ic "import pfind"

# Package-related stuff
test:
	poetry run pytest
clean:
	find . -type d -name ".pytest_cache" | xargs rm -rf
	find . -type d -name "__pycache__" | xargs rm -rf
	rm -f ${DIR}/freqcd

# Python package management
install: install-poetry
	poetry install
install-poetry:
	pip install -U poetry
uninstall:
	poetry run pip uninstall ${PACKAGE}

# Read timestamp as hex / binary (little-endian 32-bit words)
read:
	@-xxd -e -g 4 -c 8 $(READ_ARGS) | cut -b11-27 |\
	   	xxd -r -p - | xxd -b -g 4 -c 8 | cut -b1-75
readx:
	@-xxd -e -g 4 -c 8 $(READ_ARGS) | cut -b1-27

