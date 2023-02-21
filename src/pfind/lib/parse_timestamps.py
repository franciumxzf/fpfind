#!/usr/bin/env python3
# Justin, 2022-10-07
# Take in any set of timestamp data, and converts it
#
# This is for timestamp7 data, i.e. 4ps data
# -a0 is hex data, -a1 is binary, -a2 binary text
# intermediate data: t, d
#
# Sample recipes:
#
#   Split timestamp data on same channel into two different timestamp files:
#     >>> t, p = read_a1("data/0_individualdata/singletimestampnodelay.a1.dat", legacy=True)
#     >>> write_a1("data/1_rawevents/raw_alice_202210071600", t[p==1], p[p==1], legacy=True)
#     >>> write_a1("data/1_rawevents/raw_bob_202210071600", t[p==8], p[p==8], legacy=True)
#     # echo "202210071600" >> ../data/1_rawevents/raw_time_list
#
# Changelog:
#   2022-10-07 Currently does not deal with dummy events and blinding events, all set to 0

import argparse
import struct
import sys

import matplotlib.pyplot as plt
import numpy as np

TIMESTAMP_RESOLUTION = 256  # units of 1/ns
    
def read_a0(
        filename: str,
        legacy: bool = None,
        float: bool = True,
        raw: bool = False,
    ):
    """Converts a0 timestamp format into timestamps and detector pattern.
    
    If 'float' is True, then the output is returned as a 128-bit floating
    point value in fractional units of 1 ns. Otherwise, 'float' is False
    will return 64-bit integer values in multiples of 1 ns (the fractional
    component is discarded).

    If 'raw' is True, timestamps are stored in units of 4ps instead of 1ns.

    Note:
        128-bit floating point is required since 64-bit float only has a
        precision of 53-bits (timestamps have precision of 54-bits). If
        the bandwidth cost is too high when using 128-bit floats, then
        chances are the application also does not require sub-ns precision.

        Consider adding an option to return the raw timestamps in units
        of 4ps directly, so no accuracy loss is expected.
    """
    data = np.genfromtxt(filename, delimiter="\n", dtype="U8")
    data = np.array([int(v,16) for v in data]).reshape(-1, 2)
    t = ((np.uint64(data[:, 1]) << 22) + (data[:, 0] >> 10))
    if float:
        t = np.array(t, dtype=np.float128)
        if not raw:
            t = t / TIMESTAMP_RESOLUTION
    elif not raw:
        t = t // TIMESTAMP_RESOLUTION  # convert to units of ns
    p = data[:, 0] & 0xF
    return t, p

def read_a1(
        filename: str,
        legacy: bool = False,
        float: bool = True,
        raw: bool = False,
    ):
    high_pos = 1; low_pos = 0
    if legacy: high_pos, low_pos = low_pos, high_pos
    with open(filename, "rb") as f:
        data = np.fromfile(file=f, dtype="=I").reshape(-1, 2)
    t = ((np.uint64(data[:, high_pos]) << 22) + (data[:, low_pos] >> 10))
    if float:
        t = np.array(t, dtype=np.float128)
        if not raw:
            t = t / TIMESTAMP_RESOLUTION
    elif not raw:
        t = t // TIMESTAMP_RESOLUTION  # convert to units of ns
    p = data[:, low_pos] & 0xF
    return t, p

def read_a2(
        filename: str,
        legacy: bool = None,
        float: bool = True,
        raw: bool = False,
    ):
    data = np.genfromtxt(filename, delimiter="\n", dtype="U16")
    data = np.array([int(v,16) for v in data])
    t = np.uint64(data >> 10)
    if float:
        t = np.array(t, dtype=np.float128)
        if not raw:
            t = t / TIMESTAMP_RESOLUTION
    elif not raw:
        t = t // TIMESTAMP_RESOLUTION  # convert to units of ns
    p = data & 0xF
    return t, p

def _consolidate_events(t: list, p: list):
    # float128 is needed, since float64 only encodes 53-bits of precision,
    # while the high resolution timestamp has 54-bits precision
    # TODO(Justin, 2023-02-21):
    #   Check behaviour of code when more than 64-bit precision floating point is supplied.
    data = (np.array(t, dtype=np.float128) * TIMESTAMP_RESOLUTION).astype(np.uint64) << 10
    data += np.array(p).astype(np.uint64)
    return np.sort(data)

def write_a2(filename: str, t: list, p: list, legacy: bool = None):
    data = _consolidate_events(t, p)
    with open(filename, "w") as f:
        for line in data:
            f.write(f"{line:016x}\n")

def write_a0(filename: str, t: list, p: list, legacy: bool = None):
    events = _consolidate_events(t, p)
    data = np.empty((2*events.size,), dtype=np.uint32)
    data[0::2] = (events & 0xFFFFFFFF); data[1::2] = (events >> 32)
    with open(filename, "w") as f:
        for line in data:
            f.write(f"{line:08x}\n")

def write_a1(filename: str, t: list, p: list, legacy: bool = False):
    events = _consolidate_events(t, p)
    with open(filename, "wb") as f:
        for line in events:
            if legacy:
                line = int(line); line = ((line & 0xFFFFFFFF) << 32) + (line >> 32)
            f.write(struct.pack("=Q", line))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Converts between different timestamp7 formats")
    parser.add_argument("-A", choices=["0","1","2"], required=True, help="Input timestamp format")
    parser.add_argument("-X", action="store_true", help="Input legacy format")
    parser.add_argument("-a", choices=["0","1","2"], required=True, help="Output timestamp format")
    parser.add_argument("-x", action="store_true", help="Output legacy format")
    parser.add_argument("infile", help="Input timestamp file")
    parser.add_argument("outfile", help="Output timestamp file")

    # Do script only if arguments supplied
    # otherwise run as a normal script (for interactive mode)
    if len(sys.argv) > 1:
        args = parser.parse_args()

        read = [read_a0, read_a1, read_a2][int(args.A)]
        write = [write_a0, write_a1, write_a2][int(args.a)]

        t, p = read(args.infile, args.X)
        write(args.outfile, t, p, args.x)
