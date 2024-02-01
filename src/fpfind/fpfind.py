#!/usr/bin/env python3
"""Calculate frequency and time offsets between two parties.

The timestamps observed by either party can be either in the form of a
timestamp file in 'a1' format as defined in the timestamp device filespec,
or as epoch files 'T1'(t1dir)/'T2'(senddir) as defined in the qcrypto
filespec.

The time offset itself evolves over time when clock skew is present.
Here we define the reference time as the common starting time of the
observed timestamps of both parties, and the time offset defined at said
reference time.

To scan for possible precompensations, use the '--precomp-enable' and
'--precomp-ordered' flags, with INFO verbosity '-v'.

Changelog:
    2023-01-09, Justin: Refactoring from fpfind.py
    2023-01-31, Justin: Formalize interface for fpfind.py
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
from fpfind.lib.constants import (
    EPOCH_LENGTH, MAX_FCORR, NTP_MAXDELAY_NS, PeakFindingFailed,
)
from fpfind.lib.logging import get_logger, verbosity2level, set_logfile
from fpfind.lib.utils import (
    ArgparseCustomFormatter, parse_docstring_description,
    round, generate_fft, get_timing_delay_fft, slice_timestamps, get_xcorr, get_statistics,
    get_timestamp, get_first_overlapping_epoch, normalize_timestamps,
)

logger = get_logger(__name__, human_readable=True)

# Allows quick prototyping by interrupting execution right before FFT
# Trigger by running 'python3 -i -m fpfind.fpfind --config ... --experiment'.
# For internal use only.
_ENABLE_BREAKPOINT = False

# Controls learning rate, i.e. how much to decrease resolution by
RES_REFINE_FACTOR = np.sqrt(2)

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
        ats: list,
        bts: list,
        num_wraps: int,
        num_bins: int,
        resolution: float,
        target_resolution: float,
        threshold: float,
        separation_duration: float,
    ):
    """Perform the actual frequency compensation routine.

    Timestamps must already by normalized to starting time of 0ns. Whether
    this starting time is also common reference between 'ats' and 'bts' is up to
    implementation.

    Args:
        ats: Timestamps of reference side, in units of ns.
        bts: Timestamps of compensating side, in units of ns.
        num_wraps: Number of cross-correlations to overlay, usually 1.
        num_bins: Numbers of bins to use in the FFT.
        resolution: Initial resolution of cross-correlation.
        target_resolution: Target resolution desired from routine.
        threshold: Height of peak to discriminate as signal, in units of dev.
        separation_duration: Separation of cross-correlations, in units of ns.
    """
    end_time = min(ats[-1], bts[-1])
    duration = num_wraps * resolution * num_bins
    logger.debug("  Performing peak searching...")
    logger.debug("    Parameters:", extra={"details": [
        f"Bins: 2^{np.int32(np.log2(num_bins)):d}",
        f"Bin width: {resolution:.0f}ns",
        f"Number of wraps: {num_wraps:d}",
        f"Target resolution: {target_resolution}ns",
    ]})

    # Refinement loop, note resolution/duration will change during loop
    dt = 0
    f = 1
    curr_iteration = 1

    # Custom breakpoint for experimentation
    if _ENABLE_BREAKPOINT:
        import matplotlib.pyplot as plt  # for quick plotting
        a = ats; b = bts
        globals().update(locals())  # write all local variables to global scope
        raise

    while True:
        logger.debug(
            "    Iteration %s (r=%.1fns)",
            curr_iteration, resolution,
        )

        # Dynamically adjust 'num_wraps' based on current 'resolution',
        # avoids event overflow/underflow
        max_wraps = np.floor(end_time / (resolution * num_bins))
        num_wraps = np.round(duration / (resolution * num_bins))
        _duration = min(num_wraps, max_wraps) * resolution * num_bins

        # Perform cross-correlation
        logger.debug(
            "      Performing earlier xcorr (range: [0.00, %.2f]s)",
            _duration*1e-9,
        )
        ats_early = slice_timestamps(ats, 0, _duration)
        bts_early = slice_timestamps(bts, 0, _duration)
        afft = generate_fft(ats_early, num_bins, resolution)
        bfft = generate_fft(bts_early, num_bins, resolution)
        ys = get_xcorr(afft, bfft)

        # Calculate timing delay
        # TODO(2024-01-31): Add option to check other timing candidate.
        xs = np.arange(num_bins) * resolution
        dt1 = get_timing_delay_fft(ys, xs)[0]  # get smaller candidate
        sig = get_statistics(ys, resolution).significance

        # Confirm resolution on first run
        # If peak finding fails, 'dt' is not returned here:
        # guaranteed zero during first iteration
        while curr_iteration == 1:
            logger.debug(
                "        Peak: S = %.3f, dt = %sns (resolution = %.0fns)",
                sig, dt1, resolution,
            )

            # Deviation zero, due flat cross-correlation
            # Hint: Increase number of bins
            if sig == 0:
                raise PeakFindingFailed(
                    "Bin saturation  ",
                    significance=sig, resolution=resolution, dt1=dt1,
                )

            # If peak rejected, merge contiguous bins to double
            # the resolution of the peak search
            if sig >= threshold:
                logger.debug(f"          Accepted")
                break
            else:
                logger.debug(f"          Rejected")
                ys = np.sum(ys.reshape(-1, 2), axis = 1)
                resolution *= 2

            # Catch runaway resolution doubling, limited by
            if resolution > 1e4:
                raise PeakFindingFailed(
                    "Resolution OOB  ",
                    significance=sig, resolution=resolution, dt1=dt1,
                )

            # Recalculate timing delay since resolution changed
            xs = np.arange(len(ys)) * resolution
            dt1 = get_timing_delay_fft(ys, xs)[0]
            sig = get_statistics(ys, resolution).significance

        # Catch if timing delay exceeded
        if abs(dt1) > NTP_MAXDELAY_NS:
            raise PeakFindingFailed(
                "Time delay OOB  ",
                significance=sig, resolution=resolution, dt1=dt1,
            )

        # TODO(2024-01-31):
        #     Add short-circuit to ignore late xcorr
        #     if 'df' is near zero.
        # TODO(2024-01-31):
        #     If signal too low, spurious peaks may occur. Implement
        #     a mechanism to retry to obtain an alternative value.
        logger.debug(
            "      Performing later xcorr (range: [%.2f, %.2f]s)",
            separation_duration*1e-9, (separation_duration+_duration)*1e-9,
        )
        ats_late = slice_timestamps(ats, separation_duration, _duration)
        bts_late = slice_timestamps(bts, separation_duration, _duration)
        _afft = generate_fft(ats_late, num_bins, resolution)
        _bfft = generate_fft(bts_late, num_bins, resolution)
        _ys = get_xcorr(_afft, _bfft)

        # Calculate timing delay for late set of timestamps
        xs = np.arange(num_bins) * resolution
        _dt1 = get_timing_delay_fft(_ys, xs)[0]
        df1 = (_dt1 - dt1) / separation_duration

        # Some guard rails to make sure results make sense
        # Something went wrong with peak searching, to return intermediate
        # results which are likely near correct values.
        # TODO(2024-01-31): Fix this.
        buffer = 10
        threshold_dt = buffer*max(abs(dt), 1)
        threshold_df = buffer*max(abs(f-1), 1e-9)
        if curr_iteration != 1 and (
            abs(dt1)  > threshold_dt or \
            abs(_dt1) > threshold_dt or \
            abs(df1)  > threshold_df
        ):
            logger.warning(
                "      Interrupted due spurious signal:",
                extra={"details": [
                    f"early dt       = {dt1:10.0f} ns",
                    f"late dt        = {_dt1:10.0f} ns",
                    f"threshold dt   = {threshold_dt:10.0f} ns",
                    f"current df     = {df1*1e6:10.4f} ppm",
                    f"threshold df   = {threshold_df*1e6:10.4f} ppm",
            ]})
            break


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
        f  *= (1 + df1)
        logger.debug("      Calculated timing delays:", extra={"details": [
            f"early dt       = {dt1:10.0f} ns",
            f"late dt        = {_dt1:10.0f} ns",
            f"accumulated dt = {dt:10.0f} ns",
            f"current df     = {df1*1e6:10.4f} ppm",
            f"accumulated df = {(f-1)*1e6:10.4f} ppm",
        ]})

        # Throw error if compensation does not fall within bounds
        if abs(f - 1) >= MAX_FCORR:
            raise PeakFindingFailed(
                "Compensation OOB",
                significance=sig, resolution=resolution,
                dt1=dt1, dt2=_dt1, dt=dt, df=f-1,
            )

        # Stop if resolution met, otherwise refine resolution
        if resolution == target_resolution:
            break

        # Update for next iteration
        bts = (bts - dt1) / (1 + df1)
        resolution /= (separation_duration / duration / RES_REFINE_FACTOR)
        resolution = max(resolution, target_resolution)
        curr_iteration += 1

    df = f - 1
    logger.debug("      Returning results.")
    return dt, df

@profile
def fpfind(
        alice: list,
        bob: list,
        num_wraps: int,
        num_bins: int,
        resolution: float,
        target_resolution: float,
        threshold: float,
        separation_duration: float,
        precompensations: list,
    ):
    """Performs fpfind procedure.

    This is effectively a wrapper to 'time_freq' that performs the actual
    frequency compensation routine, but also includes a frequency
    precompensation step, as well as potentially other further refinements
    (currently unimplemented).

    Timestamps must already by normalized to starting time of 0ns. Whether
    this starting time is also common reference between 'ats' and 'bts' is up to
    implementation.

    Args:
        alice: Reference timestamps, in 'a1' format.
        bob: Target timestamps, in 'a1' format.
        num_wraps: Number of cross-correlations to overlay, usually 1.
        num_bins: Numbers of bins to use in the FFT.
        resolution: Initial resolution of cross-correlation.
        target_resolution: Target resolution desired from routine.
        threshold: Height of peak to discriminate as signal, in units of dev.
        separation_duration: Separation of cross-correlations, in units of ns.
        precompensations: List of precompensations to apply.
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
                num_wraps, num_bins, resolution, target_resolution,
                threshold, separation_duration,
            )
        except ValueError as e:
            logger.info(f"Peak finding failed, {df0*1e6:7.3f} ppm: {str(e)}")
            continue

        # Refine estimates, using the same recursive relations
        dt += f * dt1
        f *= (1 + df1)
        logger.info("  Applied another %.4f ppm compensation.", df1*1e6)
        logger.info("Peak finding successful")
        # TODO: Justify the good enough frequency value
        # TODO(2024-01-31): Add looping code to customize refinement steps.
        break

    # No appropriate frequency compensation found
    else:
        raise ValueError("No peak found!")  # TODO

    df = f - 1
    return dt, df


def generate_precompensations(start, stop, step, ordered=False) -> list:
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
    if ordered:
        df0s = sorted(df0s)
    return df0s


# fmt: on
def main():
    global _ENABLE_BREAKPOINT
    script_name = Path(sys.argv[0]).name
    parser = configargparse.ArgumentParser(
        default_config_files=[f"{script_name}.default.conf"],
        description=parse_docstring_description(__doc__),
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
        help=configargparse.SUPPRESS)

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
            "- Setting bit 2 removes time compensation\n"
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
        "--precomp-step", metavar="", type=float, default=0.1e-6,
        help="Specify the step value (default: 0.1ppm)")
    pgroup_precomp.add_argument(
        "--precomp-stop", metavar="", type=float, default=10e-6,
        help="Specify the max scan range, one-sided (default: 10ppm)")
    pgroup_precomp.add_argument(
        "--precomp-ordered", action="store_true",
        help="Test precompensations in increasing order (default: %(default)s)")

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
    if args.logging is not None:
        set_logfile(logger, args.logging, human_readable=True)
    logger.setLevel(verbosity2level(args.verbosity))
    logger.info("%s", args)

    # Set experimental mode
    if args.experiment:
        _ENABLE_BREAKPOINT = True

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
        logger.info("  Reading from epoch directories...")
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
    skip = args.skip_duration if _is_reading_ts else 0
    alice, bob = normalize_timestamps(alice, bob, skip=skip)
    logger.debug(
        "  Read %d and %d events from reference and compensating side.",
        len(alice), len(bob), extra={"details": [
            "Reference timing range: "
            f"[{alice[0]*1e-9:.2f}, {alice[-1]*1e-9:.2f}]s",
            "Compensating timing range: "
            f"[{bob[0]*1e-9:.2f}, {bob[-1]*1e-9:.2f}]s",
            f"(skipped {skip:.2f}s)",
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
            ordered=args.precomp_ordered,
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
        resolution=args.initial_res,
        target_resolution=args.final_res,
        threshold=args.threshold,
        separation_duration=Ts,
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
    if not (flag & 0b0100):
        output += f"{round(dt):d}\t"
    output = output.rstrip()
    print(output, file=sys.stdout)  # newline auto-added


if __name__ == "__main__":
    main()
