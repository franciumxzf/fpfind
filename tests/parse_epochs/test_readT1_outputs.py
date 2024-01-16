import pathlib

import numpy as np
import pytest

from fpfind import TSRES, get_overlap
from fpfind.lib import parse_timestamps as tparser
from fpfind.lib import parse_epochs as eparser

TESTCASES_READT1 = [
    ("data/raw_alice_bob/raw_bob_20221123165213.dat", "data/epochs/20221123165213/t1/ba2b5566"),
    ("data/epochs/20221109/c4.a1X.dat", "data/epochs/20221109/t1/ba0a0000"),
]


@pytest.mark.parametrize("raw_ts,first_epoch", TESTCASES_READT1)
def test_readT1epoch_1ns_noepoch(raw_ts, first_epoch):
    """Events of 1ns resolution with no MSB epoch information."""
    tt, tp = tparser.read_a1(
        raw_ts,
        legacy=True,
        resolution=TSRES.NS1,
        fractional=True,
    )
    et, ep = eparser.read_T1(
        first_epoch,
        full_epoch=False,
        resolution=TSRES.NS1,
        fractional=True,
    )

    # Compare only events where lengths overlap
    # (raw timestamps typically span multiple epochs)
    #
    # Note that raw timestamps do not encode MSB epoch information,
    # i.e. epoch number will cycle after 17-bits / 18.2 hours,
    # since this information is only supplied after 'chopper2' retrieves
    # computer time. This behaviour is encompassed in 'read_T1',
    # see documentation for relevant behaviour.
    overlap, (tt, tp, et, ep) = get_overlap(tt, tp, et, ep)
    assert overlap > 0

    # All bases must be correct
    assert all(tp == ep)

    # All timings must be equal
    assert all(tt == et)


@pytest.mark.parametrize("raw_ts,first_epoch", TESTCASES_READT1)
def test_readT1epoch_4ps_noepoch(raw_ts, first_epoch):
    tt, tp = tparser.read_a1(
        raw_ts,
        legacy=True,
        resolution=TSRES.PS4,
        fractional=False,
    )
    et, ep = eparser.read_T1(
        first_epoch,
        full_epoch=False,
        resolution=TSRES.PS4,
        fractional=False,
    )

    overlap, (tt, tp, et, ep) = get_overlap(tt, tp, et, ep)
    assert overlap > 0
    assert all(tp == ep)
    assert all(tt == et)


@pytest.mark.float80
@pytest.mark.parametrize("raw_ts,first_epoch", TESTCASES_READT1)
def test_readT1epoch_1ns_fullepoch(raw_ts, first_epoch):
    tt, tp = tparser.read_a1(
        raw_ts,
        legacy=True,
        resolution=TSRES.NS1,
        fractional=True,
    )
    et, ep = eparser.read_T1(
        first_epoch,
        full_epoch=True,
        resolution=TSRES.NS1,
        fractional=True,
    )
    
    # Since the full epoch is included, the test should
    # additionally perform modulo on the T1 epochs up to the
    # hardcoded 54-bit 4ps resolution of the timestamp data, which when
    # converted to ns yields 46-bits 1ns resolution, up to the floating
    # point accuracy of 80 bits float.
    overlap, (tt, tp, et, ep) = get_overlap(tt, tp, et, ep)
    et %= (1 << 46)

    assert overlap > 0
    assert all(tp == ep)
    assert tt == pytest.approx(et, abs=0.125)  # see epoch test limits for precision comments
    

@pytest.mark.parametrize("raw_ts,first_epoch", TESTCASES_READT1)
def test_readT1epoch_4ps_fullepoch(raw_ts, first_epoch):
    tt, tp = tparser.read_a1(
        raw_ts,
        legacy=True,
        resolution=TSRES.PS4,
        fractional=False,
    )
    et, ep = eparser.read_T1(
        first_epoch,
        full_epoch=True,
        resolution=TSRES.PS4,
        fractional=False,
    )
    
    # Since the full epoch is included, the test should
    # additionally perform modulo on the T1 epochs up to the
    # hardcoded 54-bit 4ps resolution of the timestamp data.
    overlap, (tt, tp, et, ep) = get_overlap(tt, tp, et, ep)
    et %= (1 << 54)

    assert overlap > 0
    assert all(tp == ep)
    assert all(tt == et)


@pytest.mark.parametrize("raw_ts,first_epoch", TESTCASES_READT1)
def test_readT1epoch_1ns_multipleepoch(raw_ts, first_epoch):
    """Events of 1ns resolution with no MSB epoch information."""
    tt, tp = tparser.read_a1(raw_ts, legacy=True)

    # Extract timestamps from epochs
    epoch = pathlib.Path(first_epoch)
    et = []; ep = []
    epoch_i = eparser.epoch2int(epoch.name)
    for i in range(5):
        curr_epoch = epoch.with_name(eparser.int2epoch(epoch_i + i))
        curr_et, curr_ep = eparser.read_T1(curr_epoch)
        et = np.append(et, curr_et)
        ep = np.append(ep, curr_ep)

    # Compare only events where lengths overlap
    # (raw timestamps typically span multiple epochs)
    #
    # Note that raw timestamps do not encode MSB epoch information,
    # i.e. epoch number will cycle after 17-bits / 18.2 hours,
    # since this information is only supplied after 'chopper2' retrieves
    # computer time. This behaviour is encompassed in 'read_T1',
    # see documentation for relevant behaviour.
    overlap, (tt, tp, et, ep) = get_overlap(tt, tp, et, ep)
    assert overlap > 0
    assert all(tp == ep)
    assert all(tt == et)