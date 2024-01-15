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
from fpfind.lib.utils import generate_fft, get_timing_delay_fft, slice_timestamps, get_xcorr, get_statistics

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
    start_time = max(ats[0], bts[0])
    duration = num_wraps * resolution * num_bins

    logger.debug(
        "Running with 2^%d bins of width %.0f ns, with start bias %.1e ns",
        np.int32(np.log2(num_bins)), resolution, start_time/1e9,
    )
    logger.debug(
        "Biased: ats = %s...%s, bts = %s...%s",
        ats[[0,1]] - start_time,
        ats[[-1]] - start_time,
        bts[[0,1]] - start_time,
        bts[[-1]] - start_time,
    )

    # Quit if df is near zero, i.e. no need to correct
    dt = 0
    df1 = 0
    f = 1
    is_peak_found = False
    while True:

        # Earlier cross-correlation
        logger.debug("Performing earlier xcorr")
        afft = generate_fft(
            slice_timestamps(ats, start_time, duration),
            num_bins, resolution)
        bfft = generate_fft(
            slice_timestamps(bts, start_time, duration),
            num_bins, resolution)
        ys = get_xcorr(afft, bfft)

        # Confirm resolution on first run
        while not is_peak_found:
            logger.debug("Performing first peak observation")
            stats = get_statistics(ys, resolution)
            logger.debug("S = %6.3f, dT = %s ns", stats.significance, dt)
            if stats.significance == 0:
                raise ValueError("Flatlined, need to increase number of bins")
            if stats.significance >= threshold:
                is_peak_found = True
                logger.debug("Peak found, with resolution %.0f ns", resolution)
                break
            
            # Increase the bin width
            ys = np.sum(ys.reshape(-1, 2), axis = 1)
            resolution *= 2
            logger.debug("Doubling resolution to %.0f ns", resolution)
            if resolution > 5e6:
                raise ValueError("Number of bins too little.")

        # Check significance, lower than threshold, likely wrong
        stats = get_statistics(ys, resolution)
        if stats.significance < threshold:
            logger.warning("TODO")
            break

        # Later cross-correlation
        logger.debug("Performing later xcorr")
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
        logger.debug(
            "dt1 = %.0f ns, and _dt1 = %.0f ns",
            dt1, _dt1,
        )

        # dt = dt / (1 + df1) + dt1  # update with old frequency. TODO: Verify.
        df1 = (_dt1 - dt1) / separation_duration
        dt = (dt + dt1) / (1 + df1)
        logger.debug(
            "Current estimation: dt = %.0f ns, df = %.3f ppm, separation_duration = %s",
            dt1, df1 * 1e6, separation_duration,
        )
        f *= 1 / (1 + df1)

        # Stop if resolution met, otherwise refine resolution
        if resolution <= target_resolution:
            break

        resolution = resolution / (separation_duration / duration / np.sqrt(2))
        bts = (bts - dt1) / (1 + df1)  # update from current run

    df = 1 / f - 1
    logger.debug("Returning: dt = %.0f ns, df = %.3f ppm",
        dt, df * 1e6,
    )
    return dt, df

@profile
def fpfind(alice, bob,
           num_wraps,
        num_bins, separation_duration, threshold, resolution,
        precompensation_max, precompensation_step):
    """Performs fpfind procedure.
    
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
    df0s = np.arange(1, int(precompensation_max // precompensation_step + 1) * 2) // 2
    df0s[::2] *= -1
    df0s = df0s.astype(np.float64) * precompensation_step
    logger.debug(
        "Prepared %d precompensation(s): %s... ppm",
        len(df0s), ",".join(map(str,df0s[:3])),
    )

    # Go through all precompensations
    for df0 in df0s:
        logger.debug("Applying %.3f ppm precompensation.", df0*1e6)

        # Apply frequency precompensation df0
        # TODO: CONTINUE
        df = df0
        dt, df1 = time_freq(alice, bob / (1 + df), num_wraps, resolution, num_bins, threshold, separation_duration)

        # Apply additional frequency compensation
        df = (1 + df) * (1 + df1) - 1
        logger.debug("Applying %.3f ppm compensation.", df*1e6)
        dt, df2 = time_freq(alice, bob / (1 + df), num_wraps, resolution, num_bins, threshold, separation_duration)

        # If frequency compensation successful, the subsequent correction
        # will be smaller. Assume next correction less than 0.2ppm.
        if abs(df2) < abs(df1) and abs(df2) < 2e-7:
            break

    # Refine frequency estimation to +/-0.1ppb
    df = (1 + df) * (1 + df2) - 1
    while abs(df2) > 1e-10:
        dt, df2 = time_freq(alice, bob / (1 + df), num_wraps, resolution, num_bins, threshold, separation_duration)
        df = (1 + df) * (1 + df2) - 1

    return dt, df


# fmt: on
def main():
    script_name = Path(sys.argv[0]).name
    parser = configargparse.ArgumentParser(
        default_config_files=[f"{script_name}.default.conf"],
        description=__doc__.partition("Changelog:")[0],
        formatter_class=ArgparseCustomFormatter,
    )

    # Remove metavariable name, by method injection. Makes for cleaner UI.
    def _add_argument(*args, **kwargs):
        kwargs.update(metavar="")
        return parser._add_argument(*args, **kwargs)

    parser._add_argument = parser.add_argument
    parser.add_argument = _add_argument

    # Disable Black formatting
    # fmt: off
    parser.add_argument(
        "--config", is_config_file_arg=True,
        help="Path to configuration file")
    parser.add_argument(
        "--save", is_write_out_config_file_arg=True,
        help="Path to configuration file for saving, then immediately exit")
    parser._add_argument(
        "--quiet", action="store_true",
        help="Suppress errors, but will not block logging")
    parser._add_argument(
        "-v", "--verbose", action="count", default=0,
        help="Specify debug verbosity, e.g. -vv for more verbosity")
    parser.add_argument(
        "-d", "--sendfiles",
        help="SENDFILES")
    parser.add_argument(
        "-D", "--t1files",
        help="T1FILES (reference)")
    parser.add_argument(
        "-t", "--target",
        help="Low-count side timestamp file")
    parser.add_argument(
        "-T", "--reference",
        help="High-count side timestamp file (reference)")
    parser.add_argument(
        "-V", "--verbosity", type=int,
        help="Specify output verbosity")
    parser.add_argument(
        "-e", "--first-epoch",
        help="Specify first overlapping epoch between the two remotes")
    
    # fpfind parameters
    parser.add_argument(
        "-k", "--num-wraps", type=int, default=1,
        help="Specify number of arrays to wrap. Default 1.")
    parser.add_argument(
        "-q", "--buffer-order", type=int, default=26,
        help="Specify FFT buffer order, N = 2**q")
    parser.add_argument(
        "-r", "--resolution", type=int, default=16,
        help="Specify desired timing resolution, in units of ns.")
    parser.add_argument(
        "-s", "--separation", type=int, default=6,
        help="Specify width of separation, in units of epochs.")
    parser.add_argument(
        "-S", "--threshold", type=float, default=6,
        help="Specify the statistical significance threshold.")
    # fmt: on
    args = parser.parse_args()
    logger.debug("%s", args)

    # Check whether options have been supplied, and print help otherwise
    args_sources = parser.get_source_to_settings_dict().keys()
    config_supplied = any(map(lambda x: x.startswith("config_file"), args_sources))
    if len(sys.argv) == 1 and not config_supplied:
        parser.print_help(sys.stderr)
        sys.exit(1)

    # first_epoch = args.first_epoch
    # skip_epoch = args.skip_epochs
    # num_of_epochs = args.num_epochs
    skip_epoch = 0
    num_of_epochs = args.resolution * (1 << args.buffer_order) / (1 << 29)
    separation_width = args.separation

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
        ta = read_a1(args.target, legacy=True)[0]
        tb = read_a1(args.reference, legacy=True)[0]
        # TODO: Parse ta[0] as epoch value then split from there
        # offset_start = skip_epoch*EPOCH_LENGTH
        # offset_end = offset_start + num_of_epochs*EPOCH_LENGTH
        # offset_start_wsep = offset_start + (separation_width*num_of_epochs)*EPOCH_LENGTH
        # offset_end_wsep = offset_start_wsep + num_of_epochs*EPOCH_LENGTH
        # print(offset_start, offset_end, offset_start_wsep, offset_end_wsep)
        # Ignore first epoch
        # ta0 = ta - ta[0]; tb0 = tb - tb[0]
        alice = ta
        # alice = ta[
        #     ((ta0 >= offset_start) & (ta0 <= offset_end)) |
        #     ((ta0 >= offset_start_wsep) & (ta0 <= offset_end_wsep))
        # ]
        bob = tb
        # bob = tb[
        #     ((tb0 >= offset_start) & (tb0 <= offset_end)) |
        #     ((tb0 >= offset_start_wsep) & (tb0 <= offset_end_wsep))
        # ]
        logger.debug("Read %d and %d events from high and low count side respectively.", len(bob), len(alice))

    else:
        logger.error("Timestamp files/epochs must be supplied with -tT/-dD")
        sys.exit(1)

    Ta = num_of_epochs * EPOCH_LENGTH
    DELTA_U_MAX = 0
    DELTA_U_STEP = 1e-6
    bob = bob * (1 + 50e-9)

    logger.debug("Triggered fpfind main routine.")
    td, fd = fpfind(
        alice, bob,
        num_wraps=args.num_wraps,
        num_bins=(1 << args.buffer_order),
        separation_duration=separation_width * Ta,  # Ts
        threshold=args.threshold,
        resolution=args.resolution,
        precompensation_step=DELTA_U_STEP,
        precompensation_max=DELTA_U_MAX,
    )
    print(fd)
    # print(f"{round(td):d}\t{round(fd * (1 << 34)):d}\n")


if __name__ == "__main__":
    main()
