import numpy as np

from fpfind import TSRES
from fpfind.lib import parse_timestamps as parser

FILEPATH = "data/epochs/20221109/c1.a1X.dat"

def test_reada0():
    pass

def test_reada1_1nsint():
    # Expected type
    t, p = parser.read_a1(
        FILEPATH, legacy=True, resolution=TSRES.NS1, fractional=False,
    )
    assert t.dtype == np.uint64
    assert p.dtype == np.uint32

def test_reada1_1nsfloat():
    t, p = parser.read_a1(
        FILEPATH, legacy=True, resolution=TSRES.NS1, fractional=True,
    )
    assert t.dtype == np.float128
    assert p.dtype == np.uint32

def test_reada1_4psint():
    t, p = parser.read_a1(
        FILEPATH, legacy=True, resolution=TSRES.PS4, fractional=False,
    )
    assert t.dtype == np.uint64
    assert p.dtype == np.uint32

def test_reada1_4psfloat():
    t, p = parser.read_a1(
        FILEPATH, legacy=True, resolution=TSRES.PS4, fractional=True,
    )
    assert t.dtype == np.float128
    assert p.dtype == np.uint32
