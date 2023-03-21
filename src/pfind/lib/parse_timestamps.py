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
import pathlib
import struct
import sys

import numpy as np

from pfind import TSRES

# Compilations of numpy that do not include support for 128-bit floats will not
# expose 'np.float128'. We map such instances directly into a 64-bit float instead.
# Note that some variants implicitly map 'np.float128' to 'np.float64' as well.
np_float = np.float64
if hasattr(np, "float128"):
    np_float = np.float128

    
def read_a0(
        filename: str,
        legacy: bool = None,
        resolution: TSRES = TSRES.NS1,
        fractional: bool = True,
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
    t = _format_timestamps(t, resolution, fractional)
    p = data[:, 0] & 0xF
    return t, p

def read_a1(
        filename: str,
        legacy: bool = False,
        resolution: TSRES = TSRES.NS1,
        fractional: bool = True,
    ):
    high_pos = 1; low_pos = 0
    if legacy: high_pos, low_pos = low_pos, high_pos
    with open(filename, "rb") as f:
        data = np.fromfile(file=f, dtype="=I").reshape(-1, 2)
    t = ((np.uint64(data[:, high_pos]) << 22) + (data[:, low_pos] >> 10))
    t = _format_timestamps(t, resolution, fractional)
    p = data[:, low_pos] & 0xF
    return t, p

def read_a2(
        filename: str,
        legacy: bool = None,
        resolution: TSRES = TSRES.NS1,
        fractional: bool = True,
    ):
    data = np.genfromtxt(filename, delimiter="\n", dtype="U16")
    data = np.array([int(v,16) for v in data])
    t = np.uint64(data >> 10)
    t = _format_timestamps(t, resolution, fractional)
    p = data & 0xF
    return t, p

def _format_timestamps(t: list, resolution: TSRES, fractional: bool):
    if fractional:
        t = np.array(t, dtype=np_float)
        t = t / (TSRES.PS4.value/resolution.value)
    else:
        t = np.array(t, dtype=np.uint64)
        t = t // (TSRES.PS4.value//resolution.value)
    return t

def _consolidate_events(t: list, p: list):
    # float128 is needed, since float64 only encodes 53-bits of precision,
    # while the high resolution timestamp has 54-bits precision
    data = (np.array(t, dtype=np_float) * TSRES.PS4.value).astype(np.uint64) << 10
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

def print_statistics(filename: str, t: list, p: list):
    print(f"Name: {str(filename)}")
    if pathlib.Path(filename).is_file():
        print(f"Filesize (MB): {pathlib.Path(filename).stat().st_size/(1 << 20):.3f}")
    width = 0
    if len(t) != 0:
        width = int(np.floor(np.log10(len(t)))) + 1
    print(    f"Total events    : {len(t):>{width}d}")
    if len(t) != 0:
        count = np.count_nonzero
        print(f"  Channel 1     : {count(p & 0b0001 != 0):>{width}d}")
        print(f"  Channel 2     : {count(p & 0b0010 != 0):>{width}d}")
        print(f"  Channel 3     : {count(p & 0b0100 != 0):>{width}d}")
        print(f"  Channel 4     : {count(p & 0b1000 != 0):>{width}d}")
        print(f"  Multi-channel : {count(np.isin(p, (0, 1, 2, 4, 8), invert=True)):>{width}d}")
        print(f"  No channel    : {count(p == 0):>{width}d}")
        duration = (t[-1]-t[0])*1e-9
        print(f"Total duration (s): {duration:0.9f}")
        print(f"Event rate (/s): {int(len(t)//duration)}")
        print(f"Detection patterns: {sorted(np.unique(p))}")
    
def stream_a1(
        filename: str,
        legacy: bool,
        resolution: TSRES = TSRES.NS1,
        fractional: bool = True,
    ):
    """Streaming variant of 'read_a1'.

    For large timestamp datasets where either not all timestamps need
    to be loaded into memory, or only statistics need to be retrieved.
    This avoids an OOM kill.

    Where efficiency is desired and number of timestamps is small,
    'read_a1' should be preferred instead.
    
    Usage:
        >>> for t, p in stream_a1(...):
        ...     print(t, p)
    """
    # Stream statistics
    with open(filename, "rb") as f:
        while True:
            low_word = f.read(4)
            high_word = f.read(4)
            if len(high_word) == 0:
                break
            
            # Swap words for legacy format
            if legacy:
                low_word, high_word = high_word, low_word
            low_word = struct.unpack("=I", low_word)[0]
            high_word = struct.unpack("=I", high_word)[0]

            t = (high_word << 22) + (low_word >> 10)
            t = _format_timestamps(t, resolution, fractional)
            p = low_word & 0xF
            yield t, p
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Converts between different timestamp7 formats")
    parser.add_argument("-A", choices=["0","1","2"], required=True, help="Input timestamp format")
    parser.add_argument("-X", action="store_true", help="Input legacy format")
    parser.add_argument("-p", action="store_true", help="Print statistics")
    parser.add_argument("-a", choices=["0","1","2"], default="1", help="Output timestamp format")
    parser.add_argument("-x", action="store_true", help="Output legacy format")
    parser.add_argument("infile", help="Input timestamp file")
    parser.add_argument("outfile", nargs="?", const="", help="Output timestamp file")

    # Do script only if arguments supplied
    # otherwise run as a normal script (for interactive mode)
    if len(sys.argv) > 1:
        args = parser.parse_args()

        # Check outfile supplied if '-p' not supplied
        if not args.p and not args.outfile:
            raise ValueError("destination filepath must be supplied.")

        read = [read_a0, read_a1, read_a2][int(args.A)]
        write = [write_a0, write_a1, write_a2][int(args.a)]

        t, p = read(args.infile, args.X)
        if args.p:
            print_statistics(args.infile, t, p)
        else:
            write(args.outfile, t, p, args.x)
