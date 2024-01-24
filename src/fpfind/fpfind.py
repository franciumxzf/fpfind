#!/usr/bin/env python3
"""TODO

Changelog:
    2023-01-09, Justin: Refactoring from fpfind.py

"""

import inspect
import functools
import os
import sys
import time
from pathlib import Path

import configargparse
import numpy as np

from fpfind.lib.parse_timestamps import read_a1
from fpfind.lib.constants import EPOCH_LENGTH, MAX_FCORR
from fpfind.lib.logging import get_logger, verbosity2level
from fpfind.lib.utils import (
    ArgparseCustomFormatter,
    round, generate_fft, get_timing_delay_fft, slice_timestamps, get_xcorr, get_statistics,
    get_timestamp, get_first_overlapping_epoch,
)

logger = get_logger(__name__, human_readable=True)

PROFILE_LEVEL = 0
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
        global PROFILE_LEVEL
        pad = "  " * PROFILE_LEVEL
        logger.debug(
            "%sSTART PROFILING", pad,
            stacklevel=2, extra=extras,
        )
        PROFILE_LEVEL += 1

        try:
            start = time.time()
            result = f(*args, **kwargs)
        finally:
            end = time.time()
            elapsed = end - start
            logger.debug(
                "%sEND PROFILING: %.0f s", pad, elapsed,
                stacklevel=2, extra=extras,
            )
            PROFILE_LEVEL -= 1

        return result
    return wrapper


# Main algorithm
@profile
def time_freq(
        ats, bts, num_wraps, resolution, target_resolution,
        num_bins, threshold, separation_duration,
    ):
    """Perform """
    # TODO: Be careful of slice timestamps
    # BOOKKEEPING
    start_time = 0
    duration = num_wraps * resolution * num_bins

    logger.debug("  Performing peak searching...")
    logger.debug("    Parameters:", extra={"details": [
        f"Bins: 2^{np.int32(np.log2(num_bins)):d}",
        f"Bin width: {resolution:.0f}ns",
        f"Number of wraps: {num_wraps:d}",
        f"Target resolution: {target_resolution}ns",
    ]})

    # Quit if df is near zero, i.e. no need to correct
    dt = 0
    f = 1
    curr_iteration = 1
    while True:
        logger.debug(
            "    Iteration %s:", curr_iteration,
            extra={"details": [
                f"High count side timing range: [{ats[0]*1e-9:.2f}, {ats[-1]*1e-9:.2f}]s",
                f"Low count side timing range: [{bts[0]*1e-9:.2f}, {bts[-1]*1e-9:.2f}]s",
                f"Current resolution: {resolution:.0f}ns"
            ]}
        )

        # Earlier cross-correlation
        logger.debug(
            "      Performing earlier xcorr (range: [0.00, %.2f]s)",
            duration*1e-9,
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
            stats = get_statistics(ys, resolution)
            logger.debug("        Peak: S = %.3f, dT = %sns (resolution = %.0fns)", stats.significance, dt, resolution)
            if stats.significance == 0:
                raise ValueError("Flatlined, need to increase number of bins")
            if stats.significance >= threshold:
                logger.debug(f"          Accepted")
                break

            # Increase the bin width
            logger.debug(f"          Rejected")
            ys = np.sum(ys.reshape(-1, 2), axis = 1)
            resolution *= 2
            if resolution > 1e4:
                raise ValueError("Number of bins too little.")

        # Later cross-correlation
        logger.debug(
            "      Performing later xcorr (range: [%.2f, %.2f]s)",
            separation_duration*1e-9,
            (separation_duration+duration)*1e-9,
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

        # Apply recursive relations
        #   A quick proof (note dt -> t):
        #     iter 0 ->  (T - t0)/f0          = T/f0    - (t0/f0)
        #     iter 1 -> ((T - t0)/f0 - t1)/f1 = T/f0/f1 - (t0/f0/f1 + t1/f1)
        #   We want:
        #     iter n ->  (T - t)/f            = T/f     - t/f
        #   Thus:
        #     f = f0 * f1 * ... * fn
        #     t = f * (t0/f0/f1/.../fn + t1/f1/.../fn + ... + tn/fn)
        #       = t0 + t1*f0 + t2*f0*f1 + ... + tn*f0*f1*...*(fn-1)
        #   Recursive relation:
        #     f' = f * fn
        #     t' = f * tn + t,  i.e. use old value of f
        dt += f * dt1
        df1 = (_dt1 - dt1) / separation_duration
        f  *= (1 + df1)
        logger.debug("      Calculated timing delays:", extra={"details": [
            f"early dt       = {dt1:10.0f} ns",
            f"late dt        = {_dt1:10.0f} ns",
            f"accumulated dt = {dt:10.0f} ns",
            f"current df     = {df1*1e6:10.4f} ppm",
            f"accumulated df = {(f-1)*1e6:10.4f} ppm",
        ]})

        # Stop if resolution met, otherwise refine resolution
        if resolution <= target_resolution:
            break

        # Throw error if compensation does not fall within bounds
        if abs(f - 1) >= MAX_FCORR:
            raise ValueError("Compensation frequency diverged")

        # Update for next iteration
        # TODO: Short-circuit later xcorr if frequency difference is already zero, but need to be careful when array is flatlined
        bts = (bts - dt1) / (1 + df1)
        resolution = resolution / (separation_duration / duration / np.sqrt(2))
        curr_iteration += 1

    df = f - 1
    logger.debug("      Returning results.")
    return dt, df

@profile
def fpfind(
        alice, bob, num_wraps, num_bins, separation_duration,
        threshold, resolution, target_resolution, precompensations,
    ):
    """Performs fpfind procedure.

    'alice' and 'bob' must have starting timestamps zeroed. TODO.

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
    df0s = precompensations

    # Go through all precompensations
    for df0 in df0s:
        logger.debug("  Applied initial %.4f ppm precompensation.", df0*1e6)

        # Apply frequency precompensation df0
        dt = 0
        f = 1 + df0
        try:
            dt1, df1 = time_freq(
                alice, (bob - dt)/f,
                num_wraps, resolution, target_resolution, num_bins, threshold, separation_duration,
            )
        except ValueError as e:
            logger.debug("  Peak finding failed with %.4f ppm precompensation: %s", df0*1e6, e)
            continue

        # Refine estimates, using the same recursive relations
        # Try once more, with more gusto...!
        dt += f * dt1
        f *= (1 + df1)
        logger.debug("  Applied another %.4f ppm compensation.", df1*1e6)
        dt2, df2 = time_freq(
            alice, (bob - dt)/f,
            num_wraps, resolution, target_resolution, num_bins, threshold, separation_duration,
        )

        # If frequency compensation successful, the subsequent correction
        # will be smaller. Assume next correction less than 0.2ppm.
        if abs(df2) <= abs(df1) and abs(df2) < 0.2e-6:
            break

    # No appropriate frequency compensation found
    else:
        raise ValueError("No peak found!")  # TODO

    # Refine frequency estimation to +/-0.1ppb
    while True:
        dt += f * dt2
        f *= (1 + df2)
        if abs(df2) <= 1e-10:
            break

        logger.debug("  Applied another %.4f ppm compensation.", df2*1e6)
        dt2, df2 = time_freq(
            alice, (bob - dt)/f,
            num_wraps, resolution, target_resolution, num_bins, threshold, separation_duration,
        )

    df = f - 1
    return dt, df


def generate_precompensations(start, stop, step) -> list:
    """Returns set of precompensations to apply before fpfind.

    The precompensations are in alternating positive/negative to allow
    scanning, e.g. for 10ppm, the sequence of values are:
    0ppm, 10ppm, -10ppm, 20ppm, -20ppm, ...

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
        "-v", "--verbosity", action="count", default=0,
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
        "-z", "--skip-epochs", metavar="", type=int, default=0,
        help="Specify number of initial epochs to skip (default: %(default)d)")

    # fpfind parameters
    pgroup_fpfind = parser.add_argument_group("fpfind parameters")
    pgroup_fpfind.add_argument(
        "-k", "--num-wraps", metavar="", type=int, default=1,
        help="Specify number of arrays to wrap (default: %(default)d)")
    pgroup_fpfind.add_argument(
        "-q", "--buffer-order", metavar="", type=int, default=26,
        help="Specify FFT buffer order, N = 2**q (default: %(default)d)")
    pgroup_fpfind.add_argument(
        "-R", "--initial-res", metavar="", type=int, default=16,
        help="Specify initial coarse timing resolution, in units of ns (default: %(default)dns)")
    pgroup_fpfind.add_argument(
        "-r", "--final-res", metavar="", type=int, default=1,
        help="Specify desired fine timing resolution, in units of ns (default: %(default)dns)")
    pgroup_fpfind.add_argument(
        "-s", "--separation", metavar="", type=float, default=6,
        help="Specify width of separation, in units of epochs (default: %(default).1f)")
    pgroup_fpfind.add_argument(
        "-S", "--threshold", metavar="", type=float, default=6,
        help="Specify the statistical significance threshold (default: %(default).1f)")
    pgroup_fpfind.add_argument(
        "-V", "--output", metavar="", type=int, default=0, choices=range(1<<4),
        help=f"{ArgparseCustomFormatter.RAW_INDICATOR}"
            "Specify output verbosity. Results are tab-delimited (default: %(default)d).\n"
            "- Setting bit 0 inverts the freq and time compensations\n"
            "- Setting bit 1 changes freq units, from abs to 2^-34\n"
            "- Setting bit 2 adds time compensation, units of 1ns\n"
            "- Setting bit 3 changes time units, from 1ns to 1/8ns"
    )

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

    # Set logging level and log arguments
    logger.setLevel(verbosity2level(args.verbosity))
    logger.debug("%s", args)

    # Verify minimum duration has been imported
    num_bins = (1 << args.buffer_order)
    Ta = args.initial_res * num_bins * args.num_wraps
    Ts = args.separation * Ta
    minimum_duration = (args.separation + 1) * Ta
    logger.debug("Reading timestamps...", extra={"details": [
        f"Required duration: {minimum_duration*1e-9:.1f}s "
        f"(cross-corr {Ta*1e-9:.1f}s)",
    ]})

    # fmt: off

    # Obtain timestamps needed for fpfind
    #   alice: low count side - chopper - HeadT2 - sendfiles (reference)
    #   bob: high count side - chopper2 - HeadT1 - t1files
    if args.sendfiles is not None and args.t1files is not None:
        logger.debug("  Reading from epoch directories...")
        _is_reading_ts = False

        # +1 epoch specified for use as buffer for frequency compensation
        required_epochs = np.ceil(minimum_duration/EPOCH_LENGTH).astype(np.int32) + args.skip_epochs + 1

        # Automatically choose first overlapping epoch if not supplied manually
        first_epoch, available_epochs = get_first_overlapping_epoch(
            args.sendfiles, args.t1files,
            first_epoch=args.first_epoch, return_length=True,
        )
        logger.debug("  ", extra={"details": [
            f"Available: {available_epochs:d} epochs "
            f"(need {required_epochs:d})",
            f"First epoch: {first_epoch}",
        ]})
        if available_epochs < required_epochs:
            logger.warning("  Insufficient epochs")

        # Read epochs
        alice = get_timestamp(
            args.sendfiles, "T2",
            first_epoch, args.skip_epochs, required_epochs - args.skip_epochs)
        bob = get_timestamp(
            args.t1files, "T1",
            first_epoch, args.skip_epochs, required_epochs - args.skip_epochs)

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
    start_time = max(alice[0], bob[0])
    if _is_reading_ts:
        start_time += args.skip_duration * 1e9  # convert to ns
    alice = slice_timestamps(alice, start_time)
    bob = slice_timestamps(bob, start_time)
    logger.debug(
        "  Read %d and %d events from high and low count side.",
        len(bob), len(alice), extra={"details": [
            f"Low count side duration: {round((alice[-1])*1e-9, 2)}s",
            f"High count side duration: {round((bob[-1])*1e-9, 2)}s",
            f"(ignored first {start_time*1e-9:.2f}s, of which "
            f"{args.skip_duration*1e-9:.2f}s was skipped)",
        ]
    })

    # Prepare frequency pre-compensations
    precompensations = [0]
    if args.precomp_enable:
        logger.debug("Generating frequency precompensations...")
        precompensations = generate_precompensations(
            args.precomp_start,
            args.precomp_stop,
            args.precomp_step,
        )
        logger.debug(
            "  Prepared %d precompensation(s): %s... ppm",
            len(precompensations),
            ",".join(map(lambda p: f"{p*1e6:g}", precompensations[:3])),
        )

    # Start fpfind
    logger.debug("Running fpfind...")
    dt, df = fpfind(
        alice, bob,
        num_wraps=args.num_wraps,
        num_bins=num_bins,
        separation_duration=Ts,
        threshold=args.threshold,
        resolution=args.initial_res,
        target_resolution=args.final_res,
        precompensations=precompensations,
    )

    # Vary output depending output verbosity value
    # Invert results, i.e. 'target' and 'reference' timestamps swapped
    # Use if the low-count side is undergoing frequency correction
    flag = args.output
    if flag & 0b0001:
        dt = -dt * (1 + df)    # t_alice * (1 + f_alice) + t_bob = 0
        df = 1 / (1 + df) - 1  # (1 + f_alice) * (1 + f_bob) = 1
    if flag & 0b0010:
        df = f"{round(df * (1 << 34)):d}"
    if flag & 0b1000:
        dt *= 8

    output = f"{df}\t"
    if flag & 0b0100:
        output += f"{round(dt):d}\t"
    output = output.rstrip()
    print(output, file=sys.stdout)  # newline auto-added

if __name__ == "__main__":
    main()
