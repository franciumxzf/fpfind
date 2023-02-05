#!/usr/bin/env python3
# Companion script to create sample binary timestamps for freqcd.c
#   Justin, 2023-02-03
#
# Examples:
#
#   1. Generate legacy timestamps and save to file:
#
#      ./generate_freqcd_testcase.py \
#          -o .input \
#          -t 0 1000000000 2000000000 \
#          -x
#      ./freqcd -f 1000000 < .input
#
#   2. Pipe timestamps directly into freqcd
#
#      ./generate_freqcd_testcase.py -t 0 1000000000 2000000000 |\
#          ./freqcd -f 1000000 |\
#          ./freqcd -f -999933 -o .output
#
#   3. Dynamically generate timestamps via command line and save to file
#
#      ./generate_freqcd_testcase.py -o .input
#

import argparse
import struct
import sys

LEGACY = False

def get_event(timestamp, detectors: int = 0b0001):
    assert isinstance(timestamp, int) and (0 <= timestamp < (1 << 54))
    assert isinstance(detectors, int) and (0 <= detectors < 16)
    event = (timestamp << 10) + detectors
    if LEGACY:
        event = ((event & 0xFFFFFFFF) << 32) + (event >> 32)
    return struct.pack("=Q", event)

def fwrite(event, filename: str = ".freqcd.input"):
    with open(filename, "ab") as f:
        f.write(event)

def owrite(event):
    sys.stdout.buffer.write(event)
    sys.stdout.flush()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generates binary timestamp events for testing freqcd.")
    parser.add_argument("-o", help="output file, defaults to stdout stream")
    parser.add_argument("-t", type=int, nargs="+", help="list of timestamps in decimal")
    parser.add_argument("-x", action="store_true", help="legacy format")

    if len(sys.argv) > 1:
        args = parser.parse_args()
        if args.x:
            LEGACY = True

        # Modify method of writing
        write = owrite
        if args.o:
            write = lambda e: fwrite(e, args.o)
            open(args.o, "wb").close()  # truncate file

        if args.t:
            # Read and validate list of integers
            for ts in args.t:
                write(get_event(ts))
        
        else:
            # Read from stdin
            print("Enter timestamps in decimal separated by newlines.")
            print("Press Ctrl-C to stop input.")
            try:
                while True:
                    try:
                        event = get_event(int(input()))
                        write(event)
                    except ValueError:
                        pass
            except KeyboardInterrupt:
                pass
