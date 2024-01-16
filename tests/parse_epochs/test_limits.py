import pathlib

import numpy as np
import pytest

from fpfind import TSRES
from fpfind.lib.parse_epochs import read_T1, write_T1

# Temporary directory for storing auto-generated data
TMP = pathlib.Path("tests/tmp")
FULL_EPOCH = (1 << 32) - 1  # largest possible epoch, 32-bits
TIMESTAMPS = [
    # Given full 32-bit epoch, sum with remaining significand in timestamp
    # can yield the overall bit accuracy. Units of 4ps.
    # Note: float64 has 53-bit resolution, while float80 has 64-bit resolution.
    #       'full' indicates (32-bit epoch)(37-bit timestamp), while
    #       'normal' indicates (17-bit epoch)(37-bit timestamp)
    #
    #  v-------- 125PS TIMESTAMP --------v     v-- 4PS RESOLUTION
    0b_00000000_00000000_00000000_00000000_00000,  # smallest timestamp
    0b_00000000_00000000_00000000_00000000_00001,  # 37-bit /* 1 */ (normal float80, uint64, int; full int) [4ps resolution]
    0b_00000000_00000000_00000000_00000000_00010,  # 36-bit /* 2 */ (normal float64) [8ps resolution]
    0b_00000000_00000000_00000000_00000000_00100,  # 35-bit
    0b_00000000_00000000_00000000_00000000_01000,  # 34-bit
    0b_00000000_00000000_00000000_00000000_10000,  # 33-bit
    0b_00000000_00000000_00000000_00000001_00000,  # 32-bit /* 6 */ (full float80, uint64) [125ps resolution]
    0b_00000000_00000000_00000000_00000010_00000,  # 31-bit
    0b_00000000_00000000_00000000_00000100_00000,  # 30-bit
    0b_00000000_00000000_00000000_00001000_00000,  # 29-bit         [1ns resolution]
    0b_00000000_00000000_00000000_00010000_00000,  # 28-bit /* 10 */
    0b_00000000_00000000_00000000_00100000_00000,  # 27-bit
    0b_00000000_00000000_00000000_01000000_00000,  # 26-bit
    0b_00000000_00000000_00000000_10000000_00000,  # 25-bit
    0b_00000000_00000000_00000001_00000000_00000,  # 24-bit
    0b_00000000_00000000_00000010_00000000_00000,  # 23-bit /* 15 */
    0b_00000000_00000000_00000100_00000000_00000,  # 22-bit
    0b_00000000_00000000_00001000_00000000_00000,  # 21-bit /* 17 */ (full float64) [256ns resolution]
    0b_00000000_00000000_00010000_00000000_00000,  # 20-bit /* 18 */

    # Other miscellaneous timestamps
    (0xffff_ffff << 5) + 0b11111,  # largest possible timestamp in epoch
]
DETECTORS = 1

@pytest.fixture()
def epochfile():
    TMP.mkdir(mode=0o775, exist_ok=True)
    filepath = write_T1(TMP, FULL_EPOCH, TIMESTAMPS, DETECTORS)
    yield filepath
    filepath.unlink()  # finalize


EXPECTED_PARAM_RESULTS = [
    # Full epoch; Units; Floating-point?; Best timestamp index
    # e.g. index 0 is normal timestamps in 1ns units with floating-point
    pytest.param(False, TSRES.PS4,   True,  1, np.float128, marks=pytest.mark.float80),  # 4ps resolution, float80
    pytest.param(False, TSRES.PS125, True,  1, np.float128, marks=pytest.mark.float80),  # 4ps resolution, float80 (not float64)
    pytest.param(True,  TSRES.PS4,   True,  6, np.float128, marks=pytest.mark.float80),  # 125ps resolution, float80
    pytest.param(False, TSRES.PS4,   True,  2, np.float128, marks=pytest.mark.float64),  # 8ps resolution, float64
    pytest.param(False, TSRES.PS125, True,  2, np.float128, marks=pytest.mark.float64),  # 8ps resolution, float64
    pytest.param(True,  TSRES.PS4,   True,  17, np.float128, marks=pytest.mark.float64), # 256ns resolution, float64
                (False, TSRES.PS4,   False, 1, np.uint64),    # 4ps resolution, uint64
                (True,  TSRES.PS4,   False, 1, object),       # 4ps resolution, int
                (True,  TSRES.PS125, False, 6, np.uint64),    # 125ps resolution, uint64
                (True,  TSRES.NS1,   False, 9, np.uint64),    # 1ns resolution, uint64
]

@pytest.mark.parametrize("full,units,floating,index,container", EXPECTED_PARAM_RESULTS)
def test_readT1_accuracy(epochfile, full, units, floating, index, container):
    # Verify resolution limits for each readT1 output type
    # If test fails, then assumptions about current platform are wrong

    t, _ = read_T1(
        epochfile,
        full_epoch=full,
        resolution=units,
        fractional=floating,
    )

    # Verify the expected container type
    assert t.dtype == container

    # Verify precision at which events are still distinguishable
    assert t[0] != t[index]

    # Verify precision at which events are indistinguishable
    # i.e. timestamp index is the upper bound precision
    # Ignored for now since only lower bound precision is a concern
    # assert t[0] == t[index-1]
