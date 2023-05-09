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
import bisect
import pathlib
import warnings
import struct
import sys
from typing import Tuple, Optional
from collections.abc import Iterator

import numpy as np
import tqdm

from fpfind import TSRES

# Compilations of numpy that do not include support for 128-bit floats will not
# expose 'np.float128'. We map such instances directly into a 64-bit float instead.
# Note that some variants implicitly map 'np.float128' to 'np.float64' as well.
np_float = np.float64
if hasattr(np, "float128"):
    np_float = np.float128
else:
    warnings.warn(
        "128-bit floats unsupported in current numpy version, using 64-bit instead."
    )



#############
#  READERS  #
#############

def read_a0(
        filename: str,
        legacy: bool = None,
        resolution: TSRES = TSRES.NS1,
        fractional: bool = True,
    ):
    """Converts a0 timestamp format into timestamps and detector pattern.

    'legacy' is set at the point of 'readevents' invocation, and determines
    the timestamp storage format.

    'resolution' can be set to any of the available TSRES enumeration, and
    denotes the units of the output timestamps. 'fractional' determines
    whether sub-unit precisions should be preserved as well. The combination
    of 'resolution' and 'fractional' will determine the resulting output
    format of the timestamps.

    Note:
        128-bit floating point is required since 64-bit float only has a
        precision of 53-bits (timestamps have precision of 54-bits). If
        the bandwidth cost is too high when using 128-bit floats, then
        chances are the application also does not require sub-ns precision.

        There is no legacy formatting for event types a0 and a2. Preserved in
        the function signature to maintain parity with 'read_a1'.
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
    """See documentation for 'read_a0'."""
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
    """See documentation for 'read_a0'."""
    data = np.genfromtxt(filename, delimiter="\n", dtype="U16")
    data = np.array([int(v,16) for v in data])
    t = np.uint64(data >> 10)
    t = _format_timestamps(t, resolution, fractional)
    p = data & 0xF
    return t, p

def sread_a1(
        filename: str,
        legacy: bool = False,
        resolution: TSRES = TSRES.NS1,
        fractional: bool = True,
        buffer_size: int = 100_000,
    ) -> Tuple[Iterator[list,list], int]:
    """Block streaming variant of 'read_a1'.

    For large timestamp datasets where either not all timestamps need
    to be loaded into memory, or only statistics need to be retrieved.
    'buffer_size' in bytes, other arguments to follow as documented in
    'read_a0'.

    If timestamp formatting is not required, the raw timestamp resolution
    can be used to speed up computation time, i.e. 'resolution' of
    TSRES.PS4 and 'fractional' of False.

    Note that these streamer can easily be extended to perform some
    pre-processing cleanup, see Usage.

    Args:
        buffer_size: Buffer size in number of events.

    Usage:
        # Application note: Block preprocessing
        #>>> for t, p in stream_a1(...):
        #...     t, p = preprocess(t, p)
        #...     yield t, p

    Note:
        Performance of naive streaming is poor (roughly 2 orders magnitude
        slower), and deprecated thusly. A compromise, whereby blocks are
        streamed instead, yields minimal performance loss while avoiding
        issues with memory (to avoid an OOM kill).

        A buffer size between 100kB and 1MB is optimal. Disk read latencies
        dominate at low buffer sizes, while memory allocation latencies
        dominate at high buffer sizes.

        Where efficiency is desired and number of timestamps is small,
        'read_a1' should be preferred instead.

    Usage:
        #>>> for t, p in stream_a1(...):
        #...     print(t, p)
    """
    def _sread_a1():
        high_pos = 1; low_pos = 0
        if legacy: high_pos, low_pos = low_pos, high_pos
        with open(filename, "rb") as f:
            while True:
                buffer = f.read(buffer_size*8)  # 8 bytes per event
                if len(buffer) == 0:
                    break

                data = np.frombuffer(buffer, dtype="=I").reshape(-1, 2)
                t = ((np.uint64(data[:, high_pos]) << 22) + (data[:, low_pos] >> 10))
                t = _format_timestamps(t, resolution, fractional)
                p = data[:, low_pos] & 0xF
                yield t, p
    
    size = pathlib.Path(filename).stat().st_size
    num_batches = int(((size-1) // (buffer_size*8)) + 1)
    return _sread_a1(), num_batches

def sread_a0(
        filename: str,
        legacy: bool = None,
        resolution: TSRES = TSRES.NS1,
        fractional: bool = True,
        buffer_size: int = 100_000,
    ) -> Tuple[Iterator[list,list], int]:
    """See documentation for 'sread_a1'"""
    def _sread_a0():
        with open(filename, "r") as f:
            while True:
                buffer = f.read(buffer_size*18)  # 16 char per event + 2 newlines
                if len(buffer) == 0:
                    break

                data = buffer.strip().split("\n")
                data = np.array([int(v,16) for v in data]).reshape(-1, 2)
                t = ((np.uint64(data[:, 1]) << 22) + (data[:, 0] >> 10))
                t = _format_timestamps(t, resolution, fractional)
                p = data[:, 0] & 0xF
                yield t, p

    size = pathlib.Path(filename).stat().st_size
    num_batches = int(((size-1) // (buffer_size*18)) + 1)
    return _sread_a0(), num_batches

def sread_a2(
        filename: str,
        legacy: bool = None,
        resolution: TSRES = TSRES.NS1,
        fractional: bool = True,
        buffer_size: int = 100_000,
    ) -> Tuple[Iterator[list,list], int]:
    """See documentation for 'sread_a1'"""
    def _sread_a2():
        with open(filename, "r") as f:
            while True:
                buffer = f.read(buffer_size*17)  # 16 char per event + 1 char newline
                if len(buffer) == 0:
                    break

                data = buffer.strip().split("\n")
                data = np.array([int(v,16) for v in data])
                t = np.uint64(data >> 10)
                t = _format_timestamps(t, resolution, fractional)
                p = data & 0xF
                yield t, p

    size = pathlib.Path(filename).stat().st_size
    num_batches = int(((size-1) // (buffer_size*17)) + 1)
    return _sread_a2(), num_batches

def _format_timestamps(t: list, resolution: TSRES, fractional: bool):
    """Returns conversion of timestamps into desired format and resolution.

    Note:
        Short-circuit when raw timestamps are requested is applied, i.e.
        'resolution' of TSRES.PS4 and 'fractional' is False. Avoiding computation
        across the entire array has significant time-savings, see below:

        >>> import timeit
        >>> t = np.random.randint(0, int(1e18), size=(100_000,), dtype=np.uint64)
        >>> _ = timeit.timeit(lambda: _format_timestamps(t, TSRES.PS4, False), number=10_000)
        # 430 us per loop without short-circuit
        # 989 ns per loop with short-circuit

        In practice, the number of batches to process is not particularly high,
        e.g. 1GB timestamp file corresponds to 1250 loops.
    """
    if fractional:
        t = np.array(t, dtype=np_float)
        t = t / (TSRES.PS4.value/resolution.value)
    elif resolution is not TSRES.PS4:  # short-circuit
        t = np.array(t, dtype=np.uint64)
        t = t // (TSRES.PS4.value//resolution.value)
    return t



#############
#  WRITERS  #
#############

def _consolidate_events(t: list, p: list, sort: bool = False):
    """Packs events into standard a1 timestamp format.

    If sorting is required, such as during timestamp merging or timestamp
    perturbation, set 'sort' to True. This invokes O(NlogN) quick sort.

    Args:
        t: Timestamp array, in units of TSRES.NS1.
        p: Detector pattern array.
        sort: If events should be further sorted in chronological order.
    
    Note:
        float128 is needed, since float64 only encodes 53-bits of precision,
        while the high resolution timestamp has 54-bits precision.
    """
    data = (np.array(t, dtype=np_float) * TSRES.PS4.value).astype(np.uint64) << 10
    data += np.array(p).astype(np.uint64)
    if sort:
        data = np.sort(data)
    return data

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

def swrite_a1(
        filename: str,
        stream: Iterator[Tuple],
        num_batches: Optional[int] = None,
        legacy: bool = False,
        display: bool = True,
    ):
    """Block streaming variant of 'write_a1'.

    Input stream assumed to have TSRES.NS1 resolution.

    Args:
        filename: Destination filename.
        stream: Input stream, in TSRES.NS1 resolution.
        num_batches: Number of batches in stream.
        legacy: If output timestamp format should be in legacy format.
        display: If progress bar should be displayed during write.

    Note:
        Settling on a fixed resolution allows for more consistent interface
        for filtering functions. Code runtime with filtering will take a hit,
        however - this is a future todo.
    """
    if display:
        stream = tqdm.tqdm(stream, total=num_batches)
    with open(filename, "wb") as f:
        for t, p in stream:
            events = _consolidate_events(t, p)
            for line in events:
                if legacy:
                    line = int(line); line = ((line & 0xFFFFFFFF) << 32) + (line >> 32)
                f.write(struct.pack("=Q", line))

def swrite_a0(
        filename: str,
        stream: Iterator[Tuple],
        num_batches: Optional[int] = None,
        legacy: Optional[bool] = None,
        display: bool = True,
    ):
    """See documentation for 'swrite_a1'."""
    if display:
        stream = tqdm.tqdm(stream, total=num_batches)
    with open(filename, "w") as f:
        for t, p in stream:
            events = _consolidate_events(t, p)
            data = np.empty((2*events.size,), dtype=np.uint32)
            data[0::2] = (events & 0xFFFFFFFF); data[1::2] = (events >> 32)
            for line in data:
                f.write(f"{line:08x}\n")

def swrite_a2(
        filename: str,
        stream: Iterator[Tuple],
        num_batches: Optional[int] = None,
        legacy: Optional[bool] = None,
        display: bool = True,
    ):
    """See documentation for 'swrite_a1'."""
    if display:
        stream = tqdm.tqdm(stream, total=num_batches)
    with open(filename, "w") as f:
        for t, p in stream:
            data = _consolidate_events(t, p)
            for line in data:
                f.write(f"{line:016x}\n")



##############
#  PRINTERS  #
##############

def print_statistics(filename: str, t: list, p: list):
    """Prints statistics using timestamp event readers.
    
    Note:
        Maintained only for legacy reasons to support older
        reading mechanisms, e.g. 'read_a0'.
    """
    # Collect statistics
    count = np.count_nonzero
    print_statistics_report(
        filename=filename,
        num_events=len(t),
        ch1_counts=count(p & 0b0001 != 0),
        ch2_counts=count(p & 0b0010 != 0),
        ch3_counts=count(p & 0b0100 != 0),
        ch4_counts=count(p & 0b1000 != 0),
        multi_counts=count(np.isin(p, (0, 1, 2, 4, 8), invert=True)),
        non_counts=count(p == 0),
        start_timestamp=t[0],
        end_timestamp=t[-1],
        patterns=sorted(np.unique(p)),
    )

def print_statistics_stream(
        filename: str,
        stream: Iterator[Tuple],
        num_batches: Optional[int] = None,
        resolution: TSRES = TSRES.NS1,
        display: bool = True,
    ):
    """Prints statistics using timestamp event streamers."""
    # Calculate statistics
    first_t = None; last_t = None
    num_events = 0
    count_p1 = 0; count_p2 = 0; count_p3 = 0; count_p4 = 0
    count_mp = 0; count_np = 0
    set_p = set()

    # Processs batches of events
    count = np.count_nonzero
    if display:
        stream = tqdm.tqdm(stream, total=num_batches)
    for t, p in stream:
        num_events += len(p)
        count_p1 += count(p & 0b0001 != 0)
        count_p2 += count(p & 0b0010 != 0)
        count_p3 += count(p & 0b0100 != 0)
        count_p4 += count(p & 0b1000 != 0)
        count_mp += count(np.isin(p, (0, 1, 2, 4, 8), invert=True))
        count_np += count(p == 0)
        set_p.update(np.unique(p))
        
        # Store timing only if timestamps not filtered out
        if len(t) != 0:
            if first_t is None:
                first_t = t[0] / resolution.value  # convert to nanoseconds
            last_t = t[-1] / resolution.value

    print_statistics_report(
        filename=filename,
        num_events=num_events,
        ch1_counts=count_p1,
        ch2_counts=count_p2,
        ch3_counts=count_p3,
        ch4_counts=count_p4,
        multi_counts=count_mp,
        non_counts=count_np,
        start_timestamp=first_t,
        end_timestamp=last_t,
        patterns=sorted(set_p),
    )

def print_statistics_report(
        filename: str,
        num_events: int,
        ch1_counts: Optional[int] = None,
        ch2_counts: Optional[int] = None,
        ch3_counts: Optional[int] = None,
        ch4_counts: Optional[int] = None,
        multi_counts: Optional[int] = None,
        non_counts: Optional[int] = None,
        start_timestamp: Optional[float] = None,
        end_timestamp: Optional[float] = None,
        patterns: Optional[list] = None,
    ):
    """Prints the statistics report.
    
    All optional fields must be present if 'num_events' > 0.
    """

    print(f"Name: {str(filename)}")
    if pathlib.Path(filename).is_file():
        filesize = pathlib.Path(filename).stat().st_size/(1 << 20)
        print(f"Filesize (MB): {filesize:.3f}")
    width = 0
    if num_events != 0:
        width = int(np.floor(np.log10(num_events))) + 1
    print(    f"Total events    : {num_events:>{width}d}")
    if num_events != 0:
        duration = (end_timestamp-start_timestamp)*1e-9
        print(f"  Channel 1     : {ch1_counts:>{width}d}")
        print(f"  Channel 2     : {ch2_counts:>{width}d}")
        print(f"  Channel 3     : {ch3_counts:>{width}d}")
        print(f"  Channel 4     : {ch4_counts:>{width}d}")
        print(f"  Multi-channel : {multi_counts:>{width}d}")
        print(f"  No channel    : {non_counts:>{width}d}")
        print(f"Duration (s) : {duration:15.9f}")
        print(f"  ~ start    : {start_timestamp*1e-9:>15.9f}")
        print(f"  ~ end      : {end_timestamp*1e-9:>15.9f}")
        print(f"Event rate (/s) : {int(num_events//duration)}")
        print(f"Detection patterns: {patterns}")



###############
#  FILTERING  #
###############

def get_pattern_mask(
        p: list,
        pattern: int,
        mask: bool = False,
        invert: bool = False,
    ) -> Tuple[list, list]:
    """Returns a mask with bits set where patterns match, as well as result.
    
    The function behaves differently when the pattern is either used as
    a fixed pattern (when 'mask' is False), or as a bitmask (when 'mask'
    is True).
    
    When 'mask' is False, pattern matching with select events whose detector
    patterns match the exact pattern. Inverting with 'invert' as False will
    remove these events instead (i.e. pattern blacklist).

    When 'mask' is True, only events containing any bits in the pattern will
    be selected. Inverting with 'invert' as True will remove detector bits
    corresponding to any of said bits.

    Args:
        p: List of detector patterns.
        pattern: Pattern to match.
        mask: Whether pattern should be used as a bitmask.
        invert: Whether selection rules should be inverted. See documentation.

    Examples:

        >>> p = np.array([0,1,2,3,4,5,6,7,8]); t = np.array(p)

        >>> mask1, p1 = get_pattern_mask(p, 0b0101, False, False)
        >>> list(t[mask1]) == list(p1) and list(p1) == [5]
        True

        >>> mask2, p2 = get_pattern_mask(p, 0b0101, False, True)
        >>> list(t[mask2]) == list(p2) and list(p2) == [0,1,2,3,4,6,7,8]
        True

        >>> mask3, p3 = get_pattern_mask(p, 0b0101, True, False)
        >>> list(t[mask3]) == [1,3,4,5,6,7] and list(p3) == [1,1,4,5,4,5]
        True

        >>> mask4, p4 = get_pattern_mask(p, 0b0101, True, True)
        >>> list(t[mask4]) == [0,2,3,6,7,8] and list(p4) == [0,2,2,2,2,8]
        True
    
    Note:
        The function is designed to return a bitmask instead of directly
        filtering the timestamps, to allow subsequent post-processing based
        on the bitmask, e.g. inverting the returned bitmask, or mark patterns.
    """
    
    # Fixed pattern
    if not mask:
        pmask = (p == pattern)
        if invert:
            pmask = ~pmask
        masked = p[pmask]

    # Pattern as bitmask
    else:
        if not invert:
            _p = (p & pattern)  # patterns containing bitmask bits
            pmask = (_p != 0)
            masked = _p[pmask]
        else:
            # Set bit 4 to indicate dummy events, since
            # non-events already do not contain the bit pattern
            _p = np.where(p == 0, 16, p)
            _p = _p ^ (_p & pattern)  # patterns with bitmask removed
            pmask = (_p != 0)
            masked = _p[pmask]  # return detector patterns after masking
            masked = np.where(masked == 16, 0, masked)  # recover dummy events
    
    return pmask, masked

def get_timing_mask(
        t: list,
        start: Optional[float] = None,
        end: Optional[float] = None,
    ) -> list:
    """Returns a mask where timestamps are bounded between start and end.
    
    The timing array is already assumed to be sorted, as part of the timestamp
    filespec. If 'start' or 'end' is None, the range is assumed unbounded in respective direction.
    Start and end time are in seconds (not ns since typical usecase as rough filter anyway).

    Examples:

        >>> t = np.array([2,4,5,8])

        >>> mask1 = get_timing_mask(t, 4, 8)  # range at boundaries
        >>> list(t[mask1]) == [4,5,8]
        True

        >>> mask2 = get_timing_mask(t, 3, 6)  # range outside boundaries
        >>> list(t[mask2]) == [4,5]
        True

        >>> mask3 = get_timing_mask(t, 9, 100)  # too high
        >>> list(t[mask3]) == []
        True

        >>> mask4 = get_timing_mask(t, 0, 1)  # too low
        >>> list(t[mask4]) == []
        True

        >>> mask5 = get_timing_mask(t, 5, 5)  # end is inclusive
        >>> list(t[mask5]) == [5]
        True

        >>> mask6 = get_timing_mask(t, start=3, end=None)  # only lower bound
        >>> list(t[mask6]) == [4,5,8]
        True

        >>> mask7 = get_timing_mask(t, start=None, end=4)  # only upper bound
        >>> list(t[mask7]) == [2,4]
        True
    """
    # Short-circuit if no results present
    mask = np.zeros(len(t), dtype=bool)
    if start is None:
        start = t[0]
    else:
        start *= 1e9  # convert s -> ns
    if end is None:
        end = t[-1]
    else:
        end *= 1e9  # convert s -> ns
    if len(t) == 0 or start > t[-1] or end < t[0]:
        return mask
    
    # Binary search
    i = bisect.bisect_left(t, start)
    j = bisect.bisect_right(t, end)
    mask[i:j] = True
    return mask



##########
#  MAIN  #
##########

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Converts between different timestamp7 formats")
    parser.add_argument("-A", choices=["0","1","2"], required=True, help="Input timestamp format")
    parser.add_argument("-X", action="store_true", help="Use legacy input format")
    parser.add_argument("-a", choices=["0","1","2"], default="1", help="Output timestamp format, defaults to 1")
    parser.add_argument("-x", action="store_true", help="Use legacy output format")
    parser.add_argument("-p", "--print", action="store_true", help="Print statistics")
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress progress indicators")

    # Filtering
    parser.add_argument("--pfilter-pattern", type=int, help="Pattern filtering: pattern")
    parser.add_argument("--pfilter-mask", action="store_true", help="Pattern filtering: set mask option")
    parser.add_argument("--pfilter-invert", action="store_true", help="Pattern filtering: set invert option")
    parser.add_argument("--tfilter-start", type=float, help="Time filtering: start timestamp, in seconds")
    parser.add_argument("--tfilter-end", type=float, help="Time filtering: end timestamp, in seconds")
    
    # Support for older read-write mechanisms, i.e. disable batch streaming
    parser.add_argument("--inmemory", action="store_true", help="Disable batch streaming")

    parser.add_argument("infile", help="Input timestamp file")
    parser.add_argument("outfile", nargs="?", const="", help="Output timestamp file")

    # Do script only if arguments supplied
    # otherwise run as a normal script (for interactive mode)
    if len(sys.argv) > 1:
        args = parser.parse_args()

        # Check outfile supplied if '-p' not supplied
        if not args.print and not args.outfile:
            raise ValueError("destination filepath must be supplied.")

        read = [read_a0, read_a1, read_a2][int(args.A)]
        sread = [sread_a0, sread_a1, sread_a2][int(args.A)]
        write = [write_a0, write_a1, write_a2][int(args.a)]
        swrite = [swrite_a0, swrite_a1, swrite_a2][int(args.a)]

        # Check file size
        filepath = pathlib.Path(args.infile)
        if not filepath.is_file():
            raise ValueError(f"'{args.infile}' is not a file.")

        # Define filter function for event and event stream
        has_filtering = \
            (args.pfilter_pattern is not None) \
            or (args.tfilter_start is not None) \
            or (args.tfilter_end is not None)
        
        def event_filter(t, p):
            if args.pfilter_pattern is not None:
                mask, p = get_pattern_mask(p, args.pfilter_pattern, args.pfilter_mask, args.pfilter_invert)
                t = t[mask]
            if args.tfilter_start is not None or args.tfilter_end is not None:
                mask = get_timing_mask(t, args.tfilter_start, args.tfilter_end)
                t = t[mask]
                p = p[mask]
            return t, p

        def inline_filter(stream):
            for t, p in stream:
                yield event_filter(t, p)
        
        # Use legacy read-write mechanisms
        if args.inmemory:
            t, p = read(filepath, args.X)
            t, p = event_filter(t, p)
            if args.p:
                print_statistics(filepath, t, p)
            else:
                write(args.outfile, t, p, args.x)

        # Check if printing stream first
        elif args.print:
            if has_filtering:
                stream, num_batches = sread(filepath, args.X, TSRES.NS1, True)
                stream = inline_filter(stream)  # requires timestamps in TSRES.NS1 resolution
                print_statistics_stream(filepath, stream, num_batches, resolution=TSRES.NS1, display=(not args.quiet))
            else:
                # Disable timestamp formatting to speed up reads
                stream, num_batches = sread(filepath, args.X, TSRES.PS4, False)
                print_statistics_stream(filepath, stream, num_batches, resolution=TSRES.PS4, display=(not args.quiet))

        # Write out
        else:
            stream, num_batches = sread(filepath, args.X, TSRES.NS1, True)
            stream = inline_filter(stream)
            swrite(args.outfile, stream, num_batches, args.x, display=(not args.quiet))
