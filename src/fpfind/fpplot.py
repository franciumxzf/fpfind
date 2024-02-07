#!/usr/bin/env python3
"""___MODULE_INFORMATION___

Changelog:
    2024-02-02, Justin: Init

References:
    [1]:
"""

import logging
import sys

import argparse
import matplotlib.pyplot as plt

import boiler.scriptutil
import boiler.logging
from S15lib.g2lib.g2lib import histogram

from fpfind.lib.parse_timestamps import read_a1
from fpfind.lib.utils import (
    get_first_overlapping_epoch, get_timestamp,
    normalize_timestamps, slice_timestamps,
)

_ENABLE_BREAKPOINT = False
logger = logging.getLogger(__name__)

def plotter(alice, bob, freq, time, width, save=False):
    bob = (bob - time) / (1 + freq*1e-6)
    ys, xs = histogram(alice, bob, duration=width)
    # Custom breakpoint for experimentation
    if _ENABLE_BREAKPOINT:
        globals().update(locals())  # write all local variables to global scope
        raise

    plt.plot(xs, ys)
    plt.xlabel("Delay (ns)")
    plt.ylabel("g(2)")
    if save:
        plt.savefig(save)
    plt.show()


def main():
    global _ENABLE_BREAKPOINT
    parser = boiler.scriptutil.generate_default_parser(__doc__)

    # Boilerplate
    pgroup_config = parser.add_argument_group("display/configuration")
    pgroup_config.add_argument(
        "-h", "--help", action="store_true",
        help="Show this help message and exit")
    pgroup_config.add_argument(
        "-v", "--verbosity", action="count", default=0,
        help="Specify debug verbosity, e.g. -vv for more verbosity")
    pgroup_config.add_argument(
        "-L", "--logging", metavar="",
        help="Log to file, if specified. Log level follows verbosity.")
    pgroup_config.add_argument(
        "--quiet", action="store_true",
        help="Suppress errors, but will not block logging")
    pgroup_config.add_argument(
        "--config", metavar="", is_config_file_arg=True,
        help="Path to configuration file")
    pgroup_config.add_argument(
        "--save", metavar="", is_write_out_config_file_arg=True,
        help="Path to configuration file for saving, then immediately exit")
    pgroup_config.add_argument(
        "--experiment", action="store_true",
        help=argparse.SUPPRESS)

    # Timestamp importing arguments
    pgroup_ts = parser.add_argument_group("importing timestamps")
    pgroup_ts.add_argument(
        "-t", "--reference", metavar="",
        help="Timestamp file in 'a1' format, from low-count side (reference)")
    pgroup_ts.add_argument(
        "-T", "--target", metavar="",
        help="Timestamp file in 'a1' format, from high-count side")
    pgroup_ts.add_argument(
        "-X", "--legacy", action="store_true",
        help="Parse raw timestamps in legacy mode (default: %(default)s)")
    pgroup_ts.add_argument(
        "-Z", "--skip-duration", metavar="", type=float, default=0,
        help="Specify initial duration to skip, in seconds (default: %(default)s)")

    # Epoch importing arguments
    pgroup_ep = parser.add_argument_group("importing epochs")
    pgroup_ep.add_argument(
        "-d", "--sendfiles", metavar="",
        help="SENDFILES, from low-count side (reference)")
    pgroup_ep.add_argument(
        "-D", "--t1files", metavar="",
        help="T1FILES, from high-count side")
    pgroup_ep.add_argument(
        "-e", "--first-epoch", metavar="",
        help="Specify filename of first overlapping epoch, optional")
    pgroup_ep.add_argument(
        "-n", "--num-epochs", metavar="", type=int, default=1,
        help="Specify number of epochs to import (default: %(default)d)")
    pgroup_ep.add_argument(
        "-z", "--skip-epochs", metavar="", type=int, default=0,
        help="Specify number of initial epochs to skip (default: %(default)d)")

    # Plotting parameters
    pgroup = parser.add_argument_group("plotting")
    pgroup.add_argument(
        "--freq", type=float, default=0.0,
        help="Specify clock skew, in units of ppm (default: %(default)f)")
    pgroup.add_argument(
        "--time", type=float, default=0.0,
        help="Specify time delay, in units of ns (default: %(default)f)")
    pgroup.add_argument(
        "--width", type=float, default=1000,
        help="Specify width of histogram, in units of ns (default: %(default)f)")
    pgroup.add_argument(
        "--duration", type=float,
        help="Specify duration of timestamps to use, units of s")
    pgroup.add_argument(
        "--save-plot",
        help="Specify filename to save the plot to")


    # Parse arguments and configure logging
    args = boiler.scriptutil.parse_args_or_help(parser)
    boiler.logging.set_default_handlers(logger, file=args.logging)
    boiler.logging.set_logging_level(logger, args.verbosity)
    logger.debug("%s", args)

    # Set experimental mode
    if args.experiment:
        _ENABLE_BREAKPOINT = True

    # Obtain timestamps needed for fpplot
    #   alice: low count side - chopper - HeadT2 - sendfiles (reference)
    #   bob: high count side - chopper2 - HeadT1 - t1files
    if args.sendfiles is not None and args.t1files is not None:
        logger.info("  Reading from epoch directories...")
        _is_reading_ts = False

        # Automatically choose first overlapping epoch if not supplied manually
        first_epoch, available_epochs = get_first_overlapping_epoch(
            args.sendfiles, args.t1files,
            first_epoch=args.first_epoch, return_length=True,
        )
        if available_epochs < args.num_epochs + args.skip_epochs:
            logger.warning("  Insufficient epochs")

        # Read epochs
        alice = get_timestamp(
            args.sendfiles, "T2",
            first_epoch, args.skip_epochs, args.num_epochs)
        bob = get_timestamp(
            args.t1files, "T1",
            first_epoch, args.skip_epochs, args.num_epochs)

    elif args.target is not None and args.reference is not None:
        logger.info("  Reading from timestamp files...")
        _is_reading_ts = True
        alice = read_a1(args.reference, legacy=args.legacy)[0]
        bob = read_a1(args.target, legacy=args.legacy)[0]

    else:
        logger.error("Timestamp files/epochs must be supplied with -tT/-dD")
        sys.exit(1)

    # Normalize timestamps to common time reference near start, so that
    # frequency compensation will not shift the timing difference too far
    skip = args.skip_duration if _is_reading_ts else 0
    alice, bob = normalize_timestamps(alice, bob, skip=skip)
    if args.duration is not None:
        alice = slice_timestamps(alice, 0, args.duration*1e9)
        bob = slice_timestamps(bob, 0, args.duration*1e9)

    plotter(alice, bob, args.freq, args.time, args.width, save=args.save_plot)


if __name__ == "__main__":
    main()
