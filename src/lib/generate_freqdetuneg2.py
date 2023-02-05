#!/usr/bin/env python3
# Justin, 2022-10-07
# From single timestamp g(2) file, generate increasingly frequency detuned timestamping.
#
# Worked out the theory with Darren:
#   Given a clock rate of f Hz, and the detuning offset denoted D,
#   the detuned frequency is f + D, where D/f = detuning ratio (e.g. 1e-9).
#   
#   Here's an example to illustrate the relationship:
#      The device keeps time via the clock rate. Suppose the clock fed to it
#      is 10Hz, then the device which registers 10 clock ticks as 1 second will
#      read out 1 second. The same device with 15 Hz will then read 1.5 seconds,
#      i.e. faster clock => timestamp is faster relative to actual time.
#
#   Given a detuning ratio R, the frequency fed to the device is now
#   f + R*f, so the effective timestamp has an additional (1+R) product.

import parse_timestamps as parser
import numpy as np

DATA_DIR = "./Clock-sync/20221007_freqdetuneg2"
FILE = DATA_DIR + "/singletimestampnodelay.a1.dat"

# Read file
t, p = parser.read_a1(FILE, legacy=True)

# Perform frequency detuning of timestamp clock
def detune(t: np.ndarray, p: np.ndarray, channel: int, ratio: float = 0, offset: float = 0):
    """Performs frequency detuning of timestamp clock on one channel.

    Args:
        t: Timestamp data
        p: Detector channel pattern
        channel: Choice of channel, follows detector pattern, i.e. 0b0100 for channel 3
        ratio: Frequency detuning ratio, i.e. 0 is no detuning
        offset: Performs fixed timing offset, optional
    """
    tc, pc = t[p==channel], p[p==channel]
    tc = (tc-t[0])*(1+ratio) + t[0] + offset  # assume both devices are synchronized t
    return tc, pc

parser.write_a2(DATA_DIR + "/c1_a2.dat", t[p==1], p[p==1], legacy=True)
parser.write_a1(DATA_DIR + "/c4e10.dat", *detune(t, p, 8, 1e-10), legacy=True)
parser.write_a1(DATA_DIR + "/c4e9.dat", *detune(t, p, 8, 1e-9), legacy=True)
parser.write_a1(DATA_DIR + "/c4e8.dat", *detune(t, p, 8, 1e-8), legacy=True)
parser.write_a1(DATA_DIR + "/c4e7.dat", *detune(t, p, 8, 1e-7), legacy=True)
parser.write_a1(DATA_DIR + "/c4e6.dat", *detune(t, p, 8, 1e-6), legacy=True)
parser.write_a1(DATA_DIR + "/c4e5.dat", *detune(t, p, 8, 1e-5), legacy=True)
parser.write_a1(DATA_DIR + "/c4e4.dat", *detune(t, p, 8, 1e-4), legacy=True)
parser.write_a1(DATA_DIR + "/c4e3.dat", *detune(t, p, 8, 1e-3), legacy=True)