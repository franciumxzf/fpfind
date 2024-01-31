import argparse
import logging
import pathlib
import typing
import warnings
from typing import Optional
from dataclasses import dataclass

import numpy as np
import scipy

from fpfind.lib.parse_epochs import (
    epoch2int, int2epoch, read_T1, read_T2,
)

inbuilt_round = round
def round(number, ndigits=None, sf=None, dp=None):
    """Stand-in replacement for in-built round, adapted from [1].

    Signature of in-build round is (number, ndigits=None). The 'dp' keyword
    is introduced as an alias to 'ndigits', as a counterpart to the 'sf'
    keyword representing the number of significant figures to use.

    Only one of the precision arguments, i.e. 'ndigits', 'sf', 'dp', should
    be supplied, otherwise the behaviour will be undefined.

    References:
        [1]: Original source, https://stackoverflow.com/a/48812729
    """
    # 'dp' overrides 'ndigits' keyword
    if ndigits is not None and dp is None:
        dp = ndigits

    # Assume regular rounding behaviour if 'sf' not supplied
    if sf is None:
        return inbuilt_round(number, dp)

    # Perform rounding
    intermediate = float('{:.{p}g}'.format(number, p=sf))
    if isinstance(number, int):
        return int(intermediate)  # preserve as int even when very large
    return '{:.{p}g}'.format(intermediate, p=sf)


def get_overlap(*arrays):
    """Returns right-truncated arrays of largest possible common length.

    Used for comparing timestamps of different length, e.g. raw timestamps
    vs chopper-generated individual epochs.
    """
    overlap = min(map(len, arrays))
    arrays = [a[:overlap] for a in arrays]
    return overlap, arrays

@dataclass
class PeakStatistics:
    signal: list
    background: list

    @property
    def max(self):
        if len(self.signal) == 0:
            return None
        return np.max(self.signal)

    @property
    def mean(self):
        if len(self.background) == 0:
            return None
        return np.mean(self.background)

    @property
    def stdev(self):
        if len(self.background) == 0:
            return None
        return np.std(self.background)

    @property
    def total(self):
        if len(self.signal) == 0:
            return None
        return sum(self.signal) - len(self.signal) * self.mean

    @property
    def significance(self):
        if self.stdev == 0:
            return 0
        return round((self.max - self.mean) / self.stdev, 3)

    @property
    def significance_raw(self):
        if self.stdev == 0:
            return None
        full = np.hstack(self.signal, self.background)
        return (np.max(full) - np.mean(full)) / np.std(full)

    @property
    def significance2(self):
        if self.stdev == 0:
            return None
        # Estimate stdev after grouping in bins of 'len(signal)'
        length = (len(self.background) // len(self.signal)) * len(self.signal)
        if length == 0:
            return None
        rebinned = np.sum(
            self.background[:length].reshape(-1, len(self.signal)), axis=1
        )
        stdev = np.std(rebinned)
        if stdev == 0:
            return None
        return round(self.total / stdev, 3)

    @property
    def g2(self):
        if self.mean == 0:
            return None
        return self.max / self.mean

def get_statistics(
    hist: list,
    resolution: Optional[float] = None,
    center: Optional[float] = None,
    window: float = 0.0,
):
    """Returns statistics of histogram, after performing cross-correlation.

    Args:
        hist: Timing histogram to analyze.
        resolution: Resolution of the histogram.
        center: Timing center of the peak, if known beforehand.
        window: Desired timing window width to exclude from background mean calculation.
    """
    # Fallback to simple statistics, if not other arguments supplied
    if resolution is None:
        if center is None:
            return PeakStatistics(hist, hist)
        else:
            raise ValueError("Resolution must be supplied if 'center' is supplied.")

    # Guess non-negative center bin position, assuming aligned at zero
    if center is None:
        bin_center = np.argmax(hist)
    else:
        bin_center = np.abs(center) // resolution
        if center < 0:
            bin_center = len(hist) - bin_center

    # Retrieve size of symmetrical window
    num_windowbins_onesided = int(np.ceil(window / 2 / resolution))
    bin_offset_left = max(0, bin_center - num_windowbins_onesided)
    bin_offset_right = min(len(hist), bin_center + num_windowbins_onesided)

    # Avoid tails of the cross-correlation by taking only half of the spectrum
    # Use-case when same timestamp is used to obtain histogram, resulting in deadtime
    bin_offset_left_bg = bin_offset_left // 2
    bin_offset_right_bg = (len(hist) + bin_offset_right) // 2

    # Retrieve signal
    signal = hist[bin_offset_left : bin_offset_right + 1]
    background = np.hstack(
        (
            hist[bin_offset_left_bg:bin_offset_left],
            hist[bin_offset_right + 1 : bin_offset_right_bg + 1],
        )
    )
    return PeakStatistics(signal, background)


@typing.no_type_check
def generate_fft(
    arr: list,
    num_bins: int,
    time_res: float,
):
    """Returns the FFT and frequency resolution for the set of timestamps.

    Assumes the inputs are real-valued, i.e. the FFT output is symmetrical.

    Args:
        arr: The timestamp series.
        num_bins: The number of bins in the time/frequency domain.
        bin_size: The size of each timing bin, in ns.

    Note:
        This function is technically not cacheable due to the mutability of
        np.ndarray.
    """
    if len(arr) == 0:
        raise ValueError("Array is empty!")
    bin_arr = np.bincount(
        np.int64((arr // time_res) % num_bins), minlength=num_bins
    )
    return scipy.fft.rfft(bin_arr)


def get_xcorr(afft: list, bfft: list, filter: Optional[list] = None):
    """Returns the cross-correlation.

    Note:
        The conjugation operation on an FFT is essentially a time-reversal
        operation on the original time-series data.
    """
    fft = np.conjugate(afft) * bfft
    if filter is not None:
        fft = fft * filter
    result = scipy.fft.irfft(fft)
    return np.abs(result)


def convert_histogram_fft(hist: list, time_bins: list):
    """Adjust axes to estimate position."""
    hlen = len(hist) // 2
    hist = np.hstack((hist[hlen:], hist[:hlen]))
    time_bins = np.hstack((-np.flip(time_bins[1:hlen+1]), time_bins[:hlen]))
    return hist, time_bins



def get_timing_delay_fft(hist: list, time_bins: list, include_negative: bool = False):
    """Returns the timing delay.

    Args:
        hist: Timing histogram
        time_bins: Time bins

    Example:
        >>> get_timing_delay_fft([1,3,0,1], [2,4,6,8])
        ()

    """
    # [0, 1, 2, 3]  -->  (0, -4)
    ppos = np.argmax(hist)
    ptime = time_bins[ppos]
    if ppos == 0:
        npos = 0
    else:
        npos = len(hist) - ppos
    ntime = -time_bins[npos]
    result = (ptime, ntime) if np.abs(ppos) < np.abs(npos) else (ntime, ptime)
    return result

def get_delay_at_index_fft(time_bins: list, pos: int):
    assert pos >= 0
    pos2 = len(time_bins) - pos if pos != 0 else 0  # corner case
    ptime = time_bins[pos]
    ntime = -time_bins[pos2]
    result = (ptime, ntime) if np.abs(pos) < np.abs(pos2) else (ntime, ptime)
    return result

def get_top_k_delays_fft(hist: list, time_bins: list, k: int):
    assert k >= 1
    if k == 1:
        return [get_delay_at_index_fft(time_bins, np.argmax(hist))]

    xs_raw = np.argpartition(hist, -k)[-k:]
    ys_raw = hist[xs_raw]
    sort = ys_raw.argsort()[::-1]  # descending order
    result = [(y, *get_delay_at_index_fft(time_bins, x)) for x, y in zip(xs_raw[sort], ys_raw[sort])]
    return result


def slice_timestamps(
    ts: list,
    start: float = None,
    duration: float = None,
):
    if start is not None:
        ts = ts - start  # note: 'ts -= start' is in-place
        ts = ts[ts >= 0]
    else:
        ts = ts - ts[0]
    if duration is not None:
        if duration >= ts[-1]:
            warnings.warn(
                f"Desired duration ({duration*1e-9:.3f} s) is longer "
                f"than available data ({ts[-1]*1e-9:.3f} s)")
        ts = ts[ts < duration]
    return ts


def histogram_fft(
    alice: list,
    bob: list,
    num_bins: int,
    resolution: float = 1,
    num_wraps: int = 1,
    acq_start: Optional[float] = None,
    freq_corr: float = 0.0,
    filter: Optional[list] = None,
    statistics: bool = False,
    center: Optional[float] = None,
    window: float = 0.0,
):
    """Returns the cross-correlation histogram.

    Args:
        acq_start: Starting timing, relative to first common timestamp.
        filter: Optional filter in frequency-space.
    """
    if not isinstance(num_wraps, (int, np.integer)):
        warnings.warn(
            "Number of wraps is not an integer - "
            "statistical significance will be lower."
        )

    duration = num_wraps * num_bins * resolution
    first_timestamp = max(alice[0], bob[0])
    last_timestamp = min(alice[-1], bob[-1])
    if first_timestamp + duration > last_timestamp:
        warnings.warn(
            f"Desired duration of timestamps ({duration} ns) "
            f"exceeds available data ({last_timestamp - first_timestamp} ns)."
        )

    # Normalize timestamps
    acq_start = 0 if acq_start is None else acq_start
    alice -= first_timestamp + acq_start
    bob -= first_timestamp + acq_start
    bob = bob * (1 + freq_corr)

    # Generate FFT
    afft, alen = generate_fft(alice, num_bins, resolution, 0, duration)
    bfft, blen = generate_fft(bob, num_bins, resolution, 0, duration)
    result = get_xcorr(afft, bfft, filter)
    bins = np.arange(num_bins) * resolution
    if statistics:
        return result, bins, alen, blen, get_statistics(result, resolution, center, window)
    return result, bins


# https://stackoverflow.com/a/23941599
class ArgparseCustomFormatter(argparse.HelpFormatter):

    RAW_INDICATOR = "rawtext|"

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

    def _split_lines(self, text, width):
        marker = ArgparseCustomFormatter.RAW_INDICATOR
        if text.startswith(marker):
            return text[len(marker):].splitlines()
        return super()._split_lines(text, width)


def get_first_overlapping_epoch(
        dir1, dir2, first_epoch=None, return_length=False,
    ):
    """Get epoch name of smallest overlapping epoch.

    If 'return_length' is True, the return value is a tuple of the epoch name
    and the number of continguous overlapping epochs starting from said epoch.
    """
    epochints1 = [epoch2int(fp.name) for fp in pathlib.Path(dir1).glob("*")]
    epochints2 = [epoch2int(fp.name) for fp in pathlib.Path(dir2).glob("*")]
    epochints = set(epochints1).intersection(epochints2)

    # Exclude epochs smaller than 'first_epoch', if supplied
    if first_epoch is not None:
        epochint_first = epoch2int(first_epoch)
        epochints = set([v for v in epochints if v >= epochint_first])

    # Calculate number of overlapping epochs
    if len(epochints) == 0:
        min_epoch = None
        num_epochs = 0
    else:
        min_epochint = min(epochints)
        min_epoch = int2epoch(min_epochint)
        num_epochs = 1
        while (min_epochint + num_epochs) in epochints:
            num_epochs += 1

    # Return result
    if return_length:
        return min_epoch, num_epochs
    return min_epoch


def get_timestamp(dirname, file_type, first_epoch, skip_epoch, num_of_epochs):
    epochdir = pathlib.Path(dirname)
    timestamp = np.array([], dtype=np.float128)
    for i in range(num_of_epochs):
        epoch_name = epochdir / int2epoch(epoch2int(first_epoch) + skip_epoch + i)
        reader = read_T1 if file_type == "T1" else read_T2
        timestamp = np.append(timestamp, reader(epoch_name)[0])
    return timestamp

def normalize_timestamps(*T, skip: float = 0.0, preserve_relative: bool = True):
    """Shifts timestamp arrays to reference zero.

    Args:
        T: List of timestamp arrays.
        skip: Duration to skip, in seconds.
        preserve_relative: Preserve relative durations between arrays.
    """
    if not preserve_relative:
        T = [slice_timestamps(ts) for ts in T]

    start_time = max([ts[0] for ts in T]) + skip*1e9  # units of ns
    T = [slice_timestamps(ts, start_time) for ts in T]
    return T
