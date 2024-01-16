import pytest
import pathlib
import subprocess

DIRECTORY = pathlib.Path("src/fpfind")
freqcd = DIRECTORY / "freqcd"
tester = DIRECTORY / "lib/generate_freqcd_testcase.py"

def parse_ts_base10(output):
    """Format: b'[NUM1]\\n[NUM2]\\n..."""
    return list(map(int, output.decode().strip().split("\n")))

# TODO(Justin, 2023-02-21):
#   Verify if the testcase is correct.
TESTCASES = [
    (1000000, [0, 1000000000, 2000000000], [0, 1000058207, 2000116415]),
]

@pytest.mark.parametrize("freq_offset,initial_ts,final_ts", TESTCASES)
def test_freqcd_onepass(freq_offset, initial_ts, final_ts):
    """Verify freqcd works for different timestamps and detuning."""

    # Convert decimal timestamps into raw timestamps
    p1 = subprocess.Popen(["python", str(tester), "-t", *list(map(str, initial_ts))], stdout=subprocess.PIPE)

    # Apply frequency offset
    p2 = subprocess.Popen([str(freqcd), "-f", str(freq_offset)], stdin=p1.stdout, stdout=subprocess.PIPE)

    # Parse raw timestamps back into decimal timestamps
    p3 = subprocess.Popen(["python", str(tester), "-"], stdin=p2.stdout, stdout=subprocess.PIPE)

    # Read output
    p1.stdout.close()
    p2.stdout.close()
    output = parse_ts_base10(p3.communicate()[0])

    assert output == final_ts
