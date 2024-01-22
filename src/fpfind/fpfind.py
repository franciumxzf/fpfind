#!/usr/bin/env python3
"""TODO

Changelog:
    2023-01-09, Justin: Refactoring from fpfind.py

"""

import inspect
import functools
import logging
import os
import sys
import time
from pathlib import Path

import configargparse
import numpy as np

from fpfind.lib.parse_timestamps import read_a1
from fpfind.lib.utils import round, generate_fft, get_timing_delay_fft, slice_timestamps, get_xcorr, get_statistics

from fpfind.lib.utils import (
    ArgparseCustomFormatter, LoggingCustomFormatter,
    get_timestamp,
)
from fpfind.lib.constants import (
    EPOCH_LENGTH,
)

# Setup logging mechanism
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(
        LoggingCustomFormatter(
            fmt="{asctime}\t{levelname:<7s}\t{funcName}:{lineno}\t| {message}",
            datefmt="%Y%m%d_%H%M%S",
            style="{",
        )
    )
    logger.addHandler(handler)
    logger.propagate = False

def profile(f):
    """Performs a simple timing profile.

    Modifies the logging facility to log the name and lineno of the function
    being called, instead of the usual encapsulating function.
    
    Possibly thread-safe, but may not yield the correct indentation levels
    during execution, in that situation.
    """

    # Allows actual function to be logged rather than wrapped
    caller = inspect.getframeinfo(inspect.stack()[1][0])
    extras = {
        "_funcname": f"[{f.__name__}]",
        "_filename": os.path.basename(caller.filename),
        # "_lineno": caller.lineno,  # this returns the @profile lineno instead
    }

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        logger.debug(
            "Started profiling...",
            stacklevel=2, extra=extras,
        )

        start = time.time()
        result = f(*args, **kwargs)
        end = time.time()

        elapsed = end - start
        logger.debug(
            "Completed profiling: %.0f s", elapsed,
            stacklevel=2, extra=extras,
        )
        return result

    return wrapper


# Main algorithm
@profile
def time_freq(
        ats, bts,
        num_wraps, resolution, num_bins, threshold, separation_duration,
    ):
    # TODO: Be careful of slice timestamps
    # BOOKKEEPING
    target_resolution = resolution
    start_time = 0
    duration = num_wraps * resolution * num_bins

    logger.debug(
        "Running with 2^%d bins of width %.0f ns",
        np.int32(np.log2(num_bins)), resolution,
    )

    # Quit if df is near zero, i.e. no need to correct
    dt = 0
    df1 = 0
    f = 1
    curr_iteration = 1
    while True:
        logger.debug("Current fpfind iteration = %s", curr_iteration)
        logger.debug(
            "  Alice timing range: [%s, %s]s",
            round((ats[0] - start_time) * 1e-9, 2),
            round((ats[-1] - start_time) * 1e-9, 2),
        )
        logger.debug(
            "  Bob timing range: [%s, %s]s",
            round((bts[0] - start_time) * 1e-9, 2),
            round((bts[-1] - start_time) * 1e-9, 2),
        )
        if curr_iteration == 1:
            logger.debug("  Target resolution = %.0f ns", resolution)
        else:
            logger.debug("  Current resolution = %.0f ns", resolution)

        # Earlier cross-correlation
        logger.debug(
            "  Performing earlier xcorr (range: [%s, %s]s)",
            round(0, 2),
            round((duration)*1e-9, 2),
        )

        afft = generate_fft(
            slice_timestamps(ats, start_time, duration),
            num_bins, resolution)

        bfft = generate_fft(
            slice_timestamps(bts, start_time, duration),
            num_bins, resolution)
        ys = get_xcorr(afft, bfft)

        # Confirm resolution on first run
        while curr_iteration == 1:
            logger.debug("  Performing first peak observation")
            stats = get_statistics(ys, resolution)
            logger.debug("    S = %6.3f, dT = %s ns", stats.significance, dt)
            if stats.significance == 0:
                raise ValueError("Flatlined, need to increase number of bins")
            if stats.significance >= threshold:
                logger.debug("    Peak found, with resolution %.0f ns", resolution)
                break
            
            # Increase the bin width
            ys = np.sum(ys.reshape(-1, 2), axis = 1)
            resolution *= 2
            logger.debug("    Doubling resolution to %.0f ns", resolution)
            if resolution > 5e6:
                raise ValueError("Number of bins too little.")

        # Check significance, lower than threshold, likely wrong
        stats = get_statistics(ys, resolution)
        if stats.significance < threshold:
            logger.warning("TODO")
            break

        # Later cross-correlation
        logger.debug(
            "  Performing later xcorr (range: [%s, %s]s)",
            round((separation_duration)*1e-9, 2),
            round((separation_duration+duration)*1e-9),
        )

        _afft = generate_fft(
            slice_timestamps(ats, start_time + separation_duration, duration),
            num_bins, resolution)
        _bfft = generate_fft(
            slice_timestamps(bts, start_time + separation_duration, duration),
            num_bins, resolution)
        _ys = get_xcorr(_afft, _bfft)

        # Calculate timing delay
        xs = np.arange(num_bins) * resolution
        dt1 = get_timing_delay_fft(ys, xs)[0]
        _dt1 = get_timing_delay_fft(_ys, xs)[0]
        logger.debug("  Calculate timing delays")
        logger.debug("    early dt = %.0f ns", dt1)
        logger.debug("    late  dt = %.0f ns", _dt1)

        # dt = dt / (1 + df1) + dt1  # update with old frequency. TODO: Verify.
        df1 = (_dt1 - dt1) / separation_duration
        logger.debug("    current df = %.3f ppm", df1 * 1e6)
        # TODO: Short-circuit later xcorr if frequency difference is already zero, but need to be careful when array is flatlined
        dt = (dt + dt1) / (1 + df1)
        f *= 1 / (1 + df1)

        # Stop if resolution met, otherwise refine resolution
        if resolution <= target_resolution:
            break

        resolution = resolution / (separation_duration / duration / np.sqrt(2))
        bts = (bts - dt1) / (1 + df1)  # update from current run
        curr_iteration += 1

    df = 1 / f - 1
    logger.debug("Final calculated result:")
    logger.debug("  dt = %.0f ns", dt)
    logger.debug("  df = %.3f ppm", df * 1e6)
    return dt, df

@profile
def fpfind(alice, bob,
           num_wraps,
        num_bins, separation_duration, threshold, resolution,
        precompensations):
    """Performs fpfind procedure.

    'alice' and 'bob' must have starting timestamps zeroed.
    
    Args:
        alice: Reference timestamps, in 'a1X' format.
        bob: Target timestamps, in 'a1X' format.
        duration: Acquisition duration.
        num_bins: Number of FFT bins.
        separation_duration: Ts
        resolution: Timing resolution, in ns.
        precompensation_max:
        precompensation_step:
    """
    # Generating frequency precompensation values,
    # e.g. for 10ppm, the sequence of values are:
    # 0ppm, 10ppm, -10ppm, 20ppm, -20ppm, ...
    df0s = precompensations

    # Go through all precompensations
    for df0 in df0s:
        logger.debug("Applying %.3f ppm precompensation.", df0*1e6)

        # Apply frequency precompensation df0
        # TODO: CONTINUE
        df = df0
        # TODO: The frequency compensation needs to be normalized to start time before applying! Better I set the reference all the way in the beginning, then no need to keep slicing up the timestamps.
        dt, df1 = time_freq(alice, bob / (1 + df), num_wraps, resolution, num_bins, threshold, separation_duration)

        # Apply additional frequency compensation
        df = (1 + df) * (1 + df1) - 1
        logger.debug("Applying %.3f ppm compensation.", df*1e6)
        dt, df2 = time_freq(alice, bob / (1 + df), num_wraps, resolution, num_bins, threshold, separation_duration)

        # If frequency compensation successful, the subsequent correction
        # will be smaller. Assume next correction less than 0.2ppm.
        if abs(df2) <= abs(df1) and abs(df2) < 2e-7:
            break

    # Refine frequency estimation to +/-0.1ppb
    df = (1 + df) * (1 + df2) - 1
    while abs(df2) > 1e-10:
        dt, df2 = time_freq(alice, bob / (1 + df), num_wraps, resolution, num_bins, threshold, separation_duration)
        df = (1 + df) * (1 + df2) - 1

    return dt, df


def generate_precompensations(start, stop, step) -> list:
    """Returns set of precompensations to apply before fpfind.

    The precompensations are in alternating positive/negative to allow
    scanning.
    
    Examples:
        >>> generate_precompensations(0, 0, 0)
        [0]
        >>> generate_precompensations(1, 10, 5)
        [1, 6, -4, 11, -9]
        >>> generate_precompensations(1, 1, 5)
        [1, 6, -4]
    """
    # Zero precompensation if invalid step supplied
    if step == 0:
        return [0]

    # Prepare pre-compensations
    df0s = np.arange(1, int(stop // step + 1) * 2) // 2
    df0s[::2] *= -1
    df0s = df0s.astype(np.float64) * step
    df0s = df0s + start
    return df0s


# fmt: on
def main():
    script_name = Path(sys.argv[0]).name
    parser = configargparse.ArgumentParser(
        default_config_files=[f"{script_name}.default.conf"],
        description=__doc__.partition("Changelog:")[0],
        formatter_class=ArgparseCustomFormatter,
        add_help=False,
    )

    # Disable Black formatting
    # fmt: off

    # Display arguments (group with defaults)
    pgroup_config = parser.add_argument_group("display/configuration")
    pgroup_config.add_argument(
        "-h", "--help", action="store_true",
        help="Show this help message and exit")
    pgroup_config.add_argument(
        "-v", "--verbose", action="count", default=0,
        help="Specify debug verbosity, e.g. -vv for more verbosity")
    pgroup_config.add_argument(
        "--quiet", action="store_true",
        help="Suppress errors, but will not block logging")
    pgroup_config.add_argument(
        "--config", metavar="", is_config_file_arg=True,
        help="Path to configuration file")
    pgroup_config.add_argument(
        "--save", metavar="", is_write_out_config_file_arg=True,
        help="Path to configuration file for saving, then immediately exit")
    
    # Timestamp importing arguments
    pgroup_ts = parser.add_argument_group("importing timestamps")
    pgroup_ts.add_argument(
        "-t", "--target", metavar="",
        help="Low-count side timestamp file")
    pgroup_ts.add_argument(
        "-T", "--reference", metavar="",
        help="High-count side timestamp file (reference)")
    pgroup_ts.add_argument(
        "-X", "--legacy", action="store_true",
        help="Parse raw timestamps in legacy mode (default: %(default)s)")
    
    # Epoch importing arguments
    pgroup_ep = parser.add_argument_group("importing epochs")
    pgroup_ep.add_argument(
        "-d", "--sendfiles", metavar="",
        help="SENDFILES")
    pgroup_ep.add_argument(
        "-D", "--t1files", metavar="",
        help="T1FILES (reference)")
    pgroup_ep.add_argument(
        "-e", "--first-epoch", metavar="",
        help="Specify first overlapping epoch between the two remotes")
    
    # fpfind parameters
    pgroup_fpfind = parser.add_argument_group("fpfind parameters")
    pgroup_fpfind.add_argument(
        "-k", "--num-wraps", metavar="", type=int, default=1,
        help="Specify number of arrays to wrap (default: %(default)d)")
    pgroup_fpfind.add_argument(
        "-q", "--buffer-order", metavar="", type=int, default=26,
        help="Specify FFT buffer order, N = 2**q (default: %(default)d)")
    pgroup_fpfind.add_argument(
        "-r", "--resolution", metavar="", type=int, default=16,
        help="Specify desired timing resolution, in units of ns (default: %(default)dns)")
    pgroup_fpfind.add_argument(
        "-s", "--separation", metavar="", type=float, default=6,
        help="Specify width of separation, in units of epochs (default: %(default).1f)")
    pgroup_fpfind.add_argument(
        "-S", "--threshold", metavar="", type=float, default=6,
        help="Specify the statistical significance threshold (default: %(default).1f)")
    pgroup_fpfind.add_argument(
        "-V", "--verbosity", metavar="", type=int, default=0,
        help="Specify output verbosity (default: %(default)d)")
    
    # Frequency pre-compensation parameters
    pgroup_precomp = parser.add_argument_group("frequency precompensation")
    pgroup_precomp.add_argument(
        "-P", "--precomp-enable", action="store_true",
        help="Enable precompensation scanning")
    pgroup_precomp.add_argument(
        "--precomp-start", metavar="", type=float, default=0.0,
        help="Specify the starting value (default: 0ppm)")
    pgroup_precomp.add_argument(
        "--precomp-step", metavar="", type=float, default=5e-6,
        help="Specify the step value (default: 5ppm)")
    pgroup_precomp.add_argument(
        "--precomp-stop", metavar="", type=float, default=100e-6,
        help="Specify the max scan range, one-sided (default: 100ppm)")

    # fmt: on
    # Parse arguments
    args = parser.parse_args()

    # Check whether options have been supplied, and print help otherwise
    args_sources = parser.get_source_to_settings_dict().keys()
    config_supplied = any(map(lambda x: x.startswith("config_file"), args_sources))
    if args.help or (len(sys.argv) == 1 and not config_supplied):
        parser.print_help(sys.stderr)
        sys.exit(1)

    # Log arguments
    logger.debug("%s", args)

    skip_epoch = 0
    Ta = args.resolution * (1 << args.buffer_order)
    num_of_epochs = Ta / (1 << 29)
    separation_width = args.separation

    logger.debug("xcorr duration = %.1f s", Ta * 1e-9)
    logger.debug("Total required duration = %.1f s", (separation_width + 1) * Ta * 1e-9)

    # fmt: off

    # alice: low count side - chopper - HeadT2 - sendfiles
    # bob: high count side - chopper2 - HeadT1 - t1files

    logger.debug("Processing timestamps")
    if args.sendfiles is not None and args.t1files is not None:
        logger.debug("Reading from directories.")
        alice = get_timestamp(args.sendfiles, 'T2', first_epoch, skip_epoch, num_of_epochs, separation_width)
        bob = get_timestamp(args.t1files, 'T1', first_epoch, skip_epoch, num_of_epochs, separation_width)

    elif args.target is not None and args.reference is not None:
        logger.debug("Reading from timestamp files.")
        alice = read_a1(args.target, legacy=args.legacy)[0]
        bob = read_a1(args.reference, legacy=args.legacy)[0]
        logger.debug("Read %d and %d events from high and low count side respectively.", len(bob), len(alice))

    else:
        logger.error("Timestamp files/epochs must be supplied with -tT/-dD")
        sys.exit(1)

    # Normalize timestamps to common time reference near start, so that
    # frequency compensation will not shift the timing difference too far
    start_time = max(alice[0], bob[0])
    alice = slice_timestamps(alice, start_time)
    bob = slice_timestamps(bob, start_time)

    # Prepare frequency pre-compensations
    precompensations = [0]
    if args.precomp_enable:
        precompensations = generate_precompensations(
            args.precomp_start,
            args.precomp_stop,
            args.precomp_step,
        )
        logger.debug(
            "Prepared %d precompensation(s): %s... ppm",
            len(precompensations),
            ",".join(map(lambda p: f"{p:g}", precompensations[:3])),
        )

    # Start fpfind
    logger.debug("Triggered fpfind main routine.")
    td, fd = fpfind(
        alice, bob,
        num_wraps=args.num_wraps,
        num_bins=(1 << args.buffer_order),
        separation_duration=separation_width * Ta,  # Ts
        threshold=args.threshold,
        resolution=args.resolution,
        precompensations=precompensations,
    )
    print(fd)
    # print(f"{round(td):d}\t{round(fd * (1 << 34)):d}\n")


if __name__ == "__main__":
    main()
