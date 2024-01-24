import pytest
import pathlib
import subprocess

DIRECTORY = pathlib.Path("src/fpfind")
freqcd = DIRECTORY / "freqcd"
tester = DIRECTORY / "lib/generate_freqcd_testcase.py"

# TODO: Formalize testcases

"""TESTCASE 1: Check freqbuffer partial buffering.

# Create pipes
mkfifo testfifo

# Start script (different shell)
./src/fpfind/freqcd -f 12300 -F testfifo

# Open file descriptors for write
exec 3>testfifo

# Pipe with multiple fcorr inputs
# Expected output: Final fcorr '34000' (after a broken '340''000' read)
python3 -c "print('n'.join(map(str, [1000*i for i in range(35)])) + 'n', end='')" | tr 'n' '\n' >> testfifo

# Teardown
exec 3>&-
"""


"""TESTCASE 2: Check fcorr updates as expected.

# Create pipes
mkfifo testfifo
mkfifo testinput

# Start script (different shell)
./src/fpfind/freqcd -i testinput -f 1000000 -F testfifo \
    | ./src/fpfind/lib/generate_freqcd_testcase.py -

# Open file descriptors for write
exec 3>testfifo
exec 4>testinput

# Pipe timestamps
# Expected output: (as per 'test_freqcd.py')
python3 ./src/fpfind/lib/generate_freqcd_testcase.py \
    -t 0 1_000_000_000 2_000_000_000 >> testinput

# Update freqcorr value
echo "0" >> testinput

# Pipe more timestamps
# Expected output: No change
python3 ./src/fpfind/lib/generate_freqcd_testcase.py \
    -t 3_000_000_000 4_000_000_000 >> testinput

# Teardown
exec 3>&-
exec 4>&-
"""