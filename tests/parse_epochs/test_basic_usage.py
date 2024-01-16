import pytest

from fpfind import get_overlap
from fpfind.lib import parse_timestamps as tparser
from fpfind.lib import parse_epochs as eparser

TESTCASES_READT1 = [
    ("data/raw_alice_bob/raw_bob_20221123165213.dat", "data/epochs/20221123165213/t1/ba2b5566"),
    ("data/epochs/20221109/c4.a1X.dat", "data/epochs/20221109/t1/ba0a0000"),
]
@pytest.mark.parametrize("raw_ts,first_epoch", TESTCASES_READT1)
def test_readT1epoch_default(raw_ts, first_epoch):
    """Events of 1ns resolution with no MSB epoch information."""
    tt, tp = tparser.read_a1(raw_ts, legacy=True)
    et, ep = eparser.read_T1(first_epoch)

    # Compare only events where lengths overlap
    overlap, (tt, tp, et, ep) = get_overlap(tt, tp, et, ep)
    assert overlap != 0

    # All bases must be correct
    assert all(tp == ep)

    # All timings must be equal, since timings follow that of timestamp device
    # TODO(Justin, 2023-02-21):
    #   Check if there is a bug in chopper2 where bitsperentry is
    #   not adhered to.
    assert all(tt == et)
