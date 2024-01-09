#!/usr/bin/env python3
"""Performs fpfind procedure.

If raw timestamps are supplied, the encoding format is expected to be 'a1X'. See qcrypto documentation for timestamp filespec.

Changelog:
    2024-01-06, Justin: Init documentation.
"""
# fmt: off

import functools
import inspect
import logging
import math
import os
import pathlib
import sys
import time
from pathlib import Path

import configargparse
import numpy as np
from scipy.fft import fft, ifft

from fpfind.lib.parse_epochs import epoch2int, int2epoch, read_T1, read_T2
from fpfind.lib.parse_timestamps import read_a1

from S15lib.g2lib.g2lib import generate_fft, get_timing_delay_fft, slice_timestamps, get_xcorr, get_statistics, histogram
from S15lib.g2lib.parse_timestamps import read_a1

DELTA_U_STEP = 1e-6 # whether to put as an option
DELTA_U_MAX = 0 #1e-4
TIMESTAMP_RESOLUTION = 256
EPOCH_LENGTH = 2 ** 29
S_th = 6 #significance limit
Ta = 2**29 #acquisition time interval/ num of periods/ epochs
Ts = 6 * Ta #separation time interval
delta_tmax = 1 # the maximum acceptable delta_t to start tracking/ timing resolution
N = 2**20 #bin number, note that this N remains unchanged during the whole algorithm/ buffer order

class CustomFormatter(logging.Formatter):
    """Supports injection of custom name overrides during logging.
    
    Copied from <https://stackoverflow.com/a/71228329>.
    """
    def format(self, record):
        if hasattr(record, "_funcname"):
            record.funcName = record._funcname
        if hasattr(record, "_filename"):
            record.filename = record._filename
        if hasattr(record, "_lineno"):
            record.lineno= record._lineno
        return super().format(record)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(
        CustomFormatter(
            fmt="{asctime}\t{levelname:<7s}\t{funcName}:{lineno}\t| {message}",
            datefmt="%Y%m%d_%H%M%S",
            style="{",
        )
    )
    logger.addHandler(handler)
    logger.propagate = False


PROFILE_INDENT = 0

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
        global PROFILE_INDENT
        indent_level = PROFILE_INDENT
        PROFILE_INDENT += 1
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
        PROFILE_INDENT -= 1
        return result

    return wrapper

@profile
def cross_corr(arr1, arr2):
    return ifft(np.conjugate(fft(arr1)) * fft(arr2))

def find_max(arr):
    arr_real = arr.real
    time_diff = 0
    max_value = arr_real[time_diff]
    time_diff = np.argmax(arr_real)
    max_value = arr_real[time_diff]
    if time_diff > (len(arr_real) / 2):
        time_diff -= len(arr_real)
    return time_diff, max_value

def statistical_significance(arr):
    ck = np.max(arr)
    ck_average = np.mean(arr)
    stdev = np.std(arr)
    if stdev == 0:
        return None
    S = (ck - ck_average) / np.std(arr)
    return S

def process_timestamp(arr, start_time, delta_t):
    new_arr = arr[np.where((arr > start_time) & (arr <= start_time + Ta))]
    bin_arr = np.zeros(N, dtype = np.int32)
    #t/delta_t mod N
    result = np.int32((new_arr // delta_t) % N)
    bin_arr[result] = 1
    return bin_arr

@profile
def time_freq(arr_a, arr_b):
    delta_t = 256 #smallest discretization time, need to make this initial dt a free variable

    T_start = max(arr_a[0], arr_b[0])
    arr_a -= T_start
    arr_b -= T_start
    T_start = 0

    #generate arrays according to equation (6)
    bin_arr_a1 = process_timestamp(arr_a, T_start, delta_t)
    bin_arr_b1 = process_timestamp(arr_b, T_start, delta_t)
    arr_c1 = cross_corr(bin_arr_a1, bin_arr_b1)

    ss = statistical_significance(arr_c1)
    if ss is None:
        raise ValueError("Cross-correlation is flat-lined: [FIXME]")
    while statistical_significance(arr_c1) <= S_th:
        if delta_t > 5e6: # if delta_t goes too large and still cannot find the peak, means need to change the hyperparameters
            raise ValueError("Cannot successfully find the peak!")
        #halve the size of the array {ck} by adding entries pairwise
        arr_c1 = np.sum(arr_c1.reshape(-1, 2), axis = 1)
        delta_t *= 2 #doubles the effective time resolution

    #After this step, we can determine the peak position from the array and the effective time resolution
    delta_T1 = find_max(arr_c1)[0] * delta_t

    #now the delta_t is already the effective time resolution
    bin_arr_a2 = process_timestamp(arr_a, T_start + Ts, delta_t)
    bin_arr_b2 = process_timestamp(arr_b, T_start + Ts, delta_t)
    arr_c2 = cross_corr(bin_arr_a2, bin_arr_b2)
    delta_T2 = find_max(arr_c2)[0] * delta_t

    delta_u = (delta_T1 - delta_T2) / Ts

    if abs(delta_u) < 1e-10:
        while delta_t > delta_tmax:
            logger.debug("dt yet to converge: %.1f --> %.1f", delta_t, delta_tmax)
            logger.debug("%s %s", Ts, Ta)
            delta_t = delta_t / (Ts / Ta / math.sqrt(2))
            new_arr_b = arr_b - delta_T1
            new_arr_a1 = process_timestamp(arr_a, T_start, delta_t)
            new_arr_b1 = process_timestamp(new_arr_b, T_start, delta_t)
            new_arr_c1 = cross_corr(new_arr_a1, new_arr_b1)

            peak_sig = statistical_significance(new_arr_c1)
            logger.debug("Peak significance: %.1f", peak_sig)
            if peak_sig < 6: # if the peak is too low, high likely the result is wrong
                break

            delta_T1_correct = find_max(new_arr_c1)[0] * delta_t
            delta_T1 += delta_T1_correct

        logger.debug("time offset: %d, no frequency offset", delta_T1)
        return delta_T1, 0

    new_arr_b = (arr_b - delta_T1) / (1 - delta_u)
    delta_T1_correct = 0
    delta_u_correct = 0
    u = 1 / (1 - delta_u)

    while delta_t > delta_tmax:
        delta_t = delta_t / (Ts / Ta / math.sqrt(2))
        new_arr_b = (new_arr_b  - delta_T1_correct) / (1 - delta_u_correct)

        new_arr_a1 = process_timestamp(arr_a, T_start, delta_t)
        new_arr_b1 = process_timestamp(new_arr_b, T_start, delta_t)
        new_arr_c1 = cross_corr(new_arr_a1, new_arr_b1)
        delta_T1_correct = find_max(new_arr_c1)[0] * delta_t

        new_arr_a2 = process_timestamp(arr_a, T_start + Ts, delta_t)
        new_arr_b2 = process_timestamp(new_arr_b, T_start + Ts, delta_t)
        new_arr_c2 = cross_corr(new_arr_a2, new_arr_b2)
        delta_T2_correct = find_max(new_arr_c2)[0] * delta_t

        delta_T1 = delta_T1_correct + (delta_T1) / (1 - delta_u_correct)
        delta_u_correct = (delta_T1_correct - delta_T2_correct) / Ts
        u *= (1 / (1 - delta_u_correct))

        if statistical_significance(new_arr_c1) < 6: # if the peak is too low, high likely the result is wrong - there might be a problem that the delta_t doesn't go to 1
            break

    delta_u = 1 / u - 1
    # print("time offset: %d, frequency offset: %f", delta_T1, delta_u)
    # logging.debug("time offset: %d, frequency offset: %f", delta_T1, delta_u)
    return delta_T1, delta_u

@profile
def fpfind(alice, bob):
    iterating_list = np.arange(1, int(DELTA_U_MAX // DELTA_U_STEP + 1) * 2) // 2
    iterating_list[::2] *= -1
    logger.debug(
        "Prepared %d precompensation(s): %s... ppm",
        len(iterating_list), ",".join(map(str,iterating_list[:3])),
    )
    for i in iterating_list:
        precomp = DELTA_U_STEP * i
        logger.debug("Applying %.3f ppm precompensation.", precomp*1e6)
        try:
            delta_T, delta_u = time_freq(alice, bob / (1 + precomp))
        except:
            continue
        print(delta_T, delta_u)
        delta_T, delta_u1 = time_freq(alice, bob / (1 + delta_u))
        if abs(delta_u1) < abs(delta_u) and abs(delta_u1) < 2e-7:
            delta_u = (1 + DELTA_U_STEP * i) * (1 + delta_u) * (1 + delta_u1) - 1
            # print(delta_T, delta_u)
            break
    while abs(delta_u1) > 1e-10:
        delta_T, delta_u1 = time_freq(alice, bob / (1 + delta_u))
        delta_u = (1 + delta_u) * (1 + delta_u1) - 1
    # logging.debug(f"time result: {delta_T}, frequency result: {delta_u}")
    return delta_T, delta_u

def result_for_freqcd(alice, bob):
    alice_copy = alice.copy()
    bob_copy = bob.copy()

    alice_freq = fpfind(bob, alice)[1]
    bob_time = fpfind(alice_copy, bob_copy)[0]

    print(f"{int(bob_time):d}\t{int(alice_freq * (1 << 34)):d}\n")

    # time result: units of 1ns
    # frequency result: units of 2e34

def get_timestamp(dir_name, file_type, first_epoch, skip_epoch, num_of_epochs, sep):
    epoch_dir = pathlib.Path(dir_name)
    timestamp = np.array([], dtype=np.float128)
    for i in range(num_of_epochs + 1):
        epoch_name = epoch_dir / int2epoch(epoch2int(first_epoch) + skip_epoch + i)
        reader = read_T1 if file_type == "T1" else read_T2
        timestamp = np.append(timestamp, reader(epoch_name)[0])  # TODO

    for i in range(num_of_epochs + 1):
        epoch_name = epoch_dir / int2epoch(epoch2int(first_epoch) + skip_epoch + sep * num_of_epochs + i)
        reader = read_T1 if file_type == "T1" else read_T2
        timestamp = np.append(timestamp, reader(epoch_name)[0])  # TODO
    return timestamp

import argparse

# https://stackoverflow.com/a/23941599
class CustomFormatter(argparse.HelpFormatter):
    def _format_action_invocation(self, action):
        if not action.option_strings:
            _ = self._metavar_formatter(action, action.dest)(1)
            print(action, _)
            metavar, = _
            return metavar
        else:
            parts = []
            # if the Optional doesn't take a value, format is:
            #    -s, --long
            if action.nargs == 0:
                parts.extend(action.option_strings)

            # if the Optional takes a value, format is:
            #    -s ARGS, --long ARGS
            # change to
            #    -s, --long ARGS
            else:
                default = action.dest.upper()
                args_string = self._format_args(action, default)
                for option_string in action.option_strings:
                    #parts.append('%s %s' % (option_string, args_string))
                    parts.append('%s' % option_string)
                parts[-1] += ' %s'%args_string
            return ', '.join(parts)


# fmt: on
def main():
    script_name = Path(sys.argv[0]).name
    parser = configargparse.ArgumentParser(
        default_config_files=[f"{script_name}.default.conf"],
        description=__doc__.partition("Changelog:")[0],
        formatter_class=CustomFormatter,
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
    parser.add_argument(
        "-n", "--num-epochs", type=int, default=1,
        help="Specify number of epochs to use")
    parser.add_argument(
        "-s", "--separation", type=int, default=6,
        help="Specify width of separation, in units of epochs.")
    parser.add_argument(
        "-q", "--buffer-order", type=int, default=20,
        help="Specify FFT buffer order, N = 2**q")
    parser.add_argument(
        "-r", "--resolution", type=int, default=1,
        help="Specify desired timing resolution, in integer units of ns.")
    parser.add_argument(
        "-S", "--threshold", type=float, default=6,
        help="Specify the statistical significance threshold.")
    parser.add_argument(
        "-k", "--skip-epochs", type=int, default=0,
        help="Specify number of skip epochs")
    # fmt: on
    args = parser.parse_args()
    logger.debug("%s", args)

    # Check whether options have been supplied, and print help otherwise
    args_sources = parser.get_source_to_settings_dict().keys()
    config_supplied = any(map(lambda x: x.startswith("config_file"), args_sources))
    if len(sys.argv) == 1 and not config_supplied:
        parser.print_help(sys.stderr)
        sys.exit(1)

    first_epoch = args.first_epoch
    skip_epoch = args.skip_epochs
    num_of_epochs = args.num_epochs
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
        offset_start = skip_epoch*EPOCH_LENGTH
        offset_end = offset_start + num_of_epochs*EPOCH_LENGTH
        offset_start_wsep = offset_start + (separation_width*num_of_epochs)*EPOCH_LENGTH
        offset_end_wsep = offset_start_wsep + num_of_epochs*EPOCH_LENGTH
        # Ignore first epoch
        ta0 = ta - ta[0]; tb0 = tb - tb[0]
        alice = ta[
            ((ta0 >= offset_start) & (ta0 <= offset_end)) |
            ((ta0 >= offset_start_wsep) & (ta0 <= offset_end_wsep))
        ]
        bob = tb[
            ((tb0 >= offset_start) & (tb0 <= offset_end)) |
            ((tb0 >= offset_start_wsep) & (tb0 <= offset_end_wsep))
        ]
        logger.debug("Read %d and %d events from high and low count side respectively.", len(bob), len(alice))

    else:
        logger.error("Timestamp files/epochs must be supplied with -tT/-dD")
        sys.exit(1)

    # TODO: Change this, avoid modifying globals...
    global S_th, Ta, Ts, delta_tmax, N
    delta_tmax = args.resolution
    Ta = num_of_epochs * EPOCH_LENGTH
    N = 2 ** args.buffer_order
    S_th = args.threshold
    Ts = separation_width * Ta

    logger.debug("Triggered fpfind main routine.")
    td, fd = fpfind(bob, alice)
    print(fd)
    # print(f"{round(td):d}\t{round(fd * (1 << 34)):d}\n")


if __name__ == "__main__":
    main()
