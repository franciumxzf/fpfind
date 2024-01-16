import pathlib

import numpy as np
import pytest

from fpfind import TSRES, get_overlap
from fpfind.lib import parse_timestamps as tparser
from fpfind.lib import parse_epochs as eparser

TESTCASES_READT2 = [
    ("data/epochs/20221109/c1.a1X.dat", "data/epochs/20221109/sendfiles/ba0a0000"),
]


@pytest.mark.parametrize("raw_ts,first_epoch", TESTCASES_READT2)
def test_readT2epoch(raw_ts, first_epoch):
    tt, tp = tparser.read_a1(
        raw_ts,
        legacy=True,
        resolution=TSRES.NS1,
        fractional=True,
    )
    et, ep = eparser.read_T2(
        first_epoch,
        full_epoch=False,
        resolution=eparser.TSRES.NS1,
        fractional=True,
    )

    # Compare only events where lengths overlap
    # (raw timestamps typically span multiple epochs)
    overlap, (tt, tp, et, ep) = get_overlap(tt, tp, et, ep)
    assert overlap > 0

    # All bases must be correct
    assert all(tp == ep)

    # All timings must be within 1 ns of each other
    # Two reasons:
    #   1. Timing information is generated in 1/256 ns resolution
    #      but stored in units of 1/8 ns in the epoch.
    #   2. Timing differences of 0 ns or 1/8 ns are manually shifted
    #      1/4 ns away, in accordance to T2 filespec
    assert tt == pytest.approx(et, abs=0.25)


@pytest.mark.parametrize("raw_ts,first_epoch", TESTCASES_READT2)
def test_readT2epoch_1ns_multipleepoch(raw_ts, first_epoch):
    """Events of 1ns resolution with no MSB epoch information."""
    tt, tp = tparser.read_a1(raw_ts, legacy=True)

    epoch = pathlib.Path(first_epoch)
    et = []; ep = []
    epoch_i = eparser.epoch2int(epoch.name)
    for i in range(3):
        curr_epoch = epoch.with_name(eparser.int2epoch(epoch_i + i))
        curr_et, curr_ep = eparser.read_T2(curr_epoch)
        et = np.append(et, curr_et)
        ep = np.append(ep, curr_ep)

    overlap, (tt, tp, et, ep) = get_overlap(tt, tp, et, ep)
    assert overlap > 0
    assert all(tp == ep)
    assert tt == pytest.approx(et, abs=0.25)
