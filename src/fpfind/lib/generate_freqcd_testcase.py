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
#   4. Read timestamps from event supplied via command line or file
#
#      cat .input | ./generate_freqcd_testcase.py
#      ./generate_freqcd_testcase.py .input
#

import argparse
import pathlib
import secrets
import struct
import sys
import warnings

import fpfind.lib.parse_timestamps as ts_parser

warnings.simplefilter(action="once", category=UserWarning)

LEGACY = False

def get_event(timestamp, detectors: int = 0b0001):
    """

    Timestamps are not restricted to the (1 << 54) cap, to reflect the
    filespec of truncating the excess significant bits. Use case in
    having user see that the timestamps have indeed overflowed, e.g. for
    verifying overflow response. A warning will be issued for precaution.
    """
    assert isinstance(timestamp, int) and (0 <= timestamp)
    if timestamp >= (1 << 54):
        warnings.warn(
            f"Timestamp overflow detected, "
            "will truncate accordingly."
        )
        timestamp = timestamp & 0x3FFFFFFFFFFFFF
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

def main():
    parser = argparse.ArgumentParser(description="Generates binary timestamp events for testing freqcd.")
    parser.add_argument("-o", help="output file, defaults to stdout stream")
    parser.add_argument("-t", type=int, nargs="+", help="list of timestamps in decimal")
    parser.add_argument("-x", action="store_true", help="legacy format")
    parser.add_argument("infile", nargs="?", help="optional filename of events ('-' if reading from stdin)")

    # Print help if no arguments supplied
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()
    if args.x:
        LEGACY = True

    # Modify method of writing
    write = owrite
    if args.o:
        write = lambda e: fwrite(e, args.o)
        open(args.o, "wb").close()  # truncate file

    # Different methods of reading timestamps
    # 1. Read decimal timestamps from command line
    #    e.g. "-t 0 1000 2000"
    if args.t:
        for ts in args.t:
            write(get_event(ts))

    # 2. Read raw timestamps from file and print as decimal
    #    e.g. "filename.dat"
    # 3. Read raw timestamps from stdin
    #    e.g. "-"
    elif args.infile:
        inputfile = args.infile

        # Direct reading from stdin currently not supported
        # instead creating a temporary file to store binary data
        HAS_TEMP_FILE = args.infile == "-"
        if HAS_TEMP_FILE:
            inputfile = pathlib.Path(secrets.token_hex(5) + "_tempinput")

        # Read binary timestamps from file
        try:
            with open(inputfile, "wb") as f:
                f.write(sys.stdin.buffer.read())

            # Parsing binary timestamps
            ts, _ = ts_parser.read_a1(inputfile, args.x)
            for t in ts:
                print(int(t * ts_parser.TSRES.PS4.value))

        # Clean up
        finally:
            if HAS_TEMP_FILE:
                inputfile.unlink()

    # 4. Read decimal timestamps from interactive prompt,
    #    note output file in this case should be supplied.
    #    This is only for testing purposes.
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

if __name__ == "__main__":
    main()
