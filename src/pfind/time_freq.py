import argparse
import numpy as np
import logging
import matplotlib.pyplot as plt
import math
import os
from scipy.fft import fft, ifft
import sys

from pfind.lib.parse_epochs import read_T1, read_T2, epoch2int, int2epoch

UPPER_LIMIT = 5e-5 # whether to put as an option
TIMESTAMP_RESOLUTION = 256
S_th = 6 #significance limit
Ta = 2**29 #acquisition time interval/ num of periods/ epochs
Ts = 6 * Ta #separation time interval
delta_tmax = 2 # the maximum acceptable delta_t to start tracking/ timing resolution
N = 2**20 #bin number, note that this N remains unchanged during the whole algorithm/ buffer order

def read_a1(filename: str, legacy: bool = False):
    high_pos = 1; low_pos = 0
    if legacy: high_pos, low_pos = low_pos, high_pos
    with open(filename, "rb") as f:
        data = np.fromfile(file=f, dtype="=I").reshape(-1, 2)
    t = ((np.uint64(data[:, high_pos]) << 22) + (data[:, low_pos] >> 10)) / TIMESTAMP_RESOLUTION
    return t

def cross_corr(arr1, arr2):
    return ifft(np.conjugate(fft(arr1)) * fft(arr2))

def find_max(arr):
    arr_real = arr.real
    time_diff = 0
    max_value = arr_real[time_diff]
    arr_sum = 0
    power_sum = 0
    for i in range(len(arr_real)):
        arr_sum += arr_real[i]
        power_sum += arr_real[i] ** 2
        if arr_real[i] > max_value:
            time_diff = i
            max_value = arr_real[i]
    if time_diff > (len(arr_real) / 2):
        time_diff -= len(arr_real)
    arr_mean = arr_sum / len(arr_real)
    sigma = math.sqrt(power_sum / len(arr_real) - arr_mean ** 2)
    return time_diff, max_value, sigma

def statistical_significance(arr):
    arr_real = arr.real
    ck = np.max(arr_real)
    ck_average = arr_real.sum() / np.size(arr_real)
    S = (ck - ck_average) / np.std(arr_real)
    return S

def process_timestamp(arr, start_time, delta_t):
    new_arr = arr[np.where((arr > start_time) & (arr <= start_time + Ta))]
    bin_arr = np.zeros(N, dtype = np.int32)
    #t/delta_t mod N
    for i in range(np.size(new_arr)):
        bin_arr[math.floor(new_arr[i] // delta_t) % N] = 1
    return bin_arr

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
            delta_t = delta_t / (Ts / Ta / math.sqrt(2))
            new_arr_b = arr_b - delta_T1
            new_arr_a1 = process_timestamp(arr_a, T_start, delta_t)
            new_arr_b1 = process_timestamp(new_arr_b, T_start, delta_t)
            new_arr_c1 = cross_corr(new_arr_a1, new_arr_b1)

            if statistical_significance(new_arr_c1) < 6: # if the peak is too low, high likely the result is wrong
                break

            delta_T1_correct = find_max(new_arr_c1)[0] * delta_t
            delta_T1 += delta_T1_correct
    
        logging.debug("time offset: %d, no frequency offset", delta_T1)
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
    logging.debug("time offset: %d, frequency offset: %f", delta_T1, delta_u)
    return delta_T1, delta_u

def pfind(alice, bob): # perform plus and minus step size concurrently
    backup_bob = bob.copy()
    freq_result = 0
    plus_freq_result = 0
    minus_freq_result = 0
    plus = True
    always_plus = False
    always_minus = False
    time_result, freq_shift1 = time_freq(alice, bob)
    plus_bob = bob.copy()
    minus_bob = bob.copy()
    new_bob = bob / (1 + freq_shift1)
    while True:
        if freq_result > 2.5e-4:
            raise ValueError("Reach algorithm limit")
            # if reach the limit still cannot obtain the correct result, means need to change the hyperparameters
        try:
            time_result, freq_shift2 = time_freq(alice, new_bob)
            if abs(freq_shift2) < 5e-7 and abs(freq_shift2) <= abs(freq_shift1): # means that the the combined offset is correct
                if not plus:
                    freq_result = plus_freq_result
                    bob = plus_bob
                elif plus:
                    freq_result = minus_freq_result
                    bob = minus_bob
                bob = bob / (1 + freq_shift1) / (1 + freq_shift2)
                freq_result = (1 + freq_result) * (1 + freq_shift1) * (1 + freq_shift2) - 1
                break
        except ValueError:
            # if cannot find the peak, go directly plus or minus the step size
            # but here we agree that our hyperparameter settings are correct
            pass
        if always_plus and always_minus:
            raise RuntimeError
        if always_plus:
            plus_bob /= (1 + UPPER_LIMIT)
            plus_freq_result = (1 + UPPER_LIMIT) * (1 + plus_freq_result) - 1
            time_result, freq_shift1 = time_freq(alice, plus_bob)
            new_bob = plus_bob / (1 + freq_shift1)
            continue
        if always_minus:
            minus_bob /= (1 - UPPER_LIMIT)
            minus_freq_result = (1 - UPPER_LIMIT) * (1 + minus_freq_result) - 1
            time_result, freq_shift1 = time_freq(alice, minus_bob)
            new_bob = minus_bob / (1 + freq_shift1)
            continue
        if plus:
            try:
                plus_bob /= (1 + UPPER_LIMIT)
                plus_freq_result = (1 + UPPER_LIMIT) * (1 + plus_freq_result) - 1
                time_result, freq_shift1 = time_freq(alice, plus_bob)
                new_bob = plus_bob / (1 + freq_shift1)
                plus = False
                continue
            except ValueError:
                always_minus = True
        if not plus:
            try:
                minus_bob /= (1 - UPPER_LIMIT)
                minus_freq_result = (1 - UPPER_LIMIT) * (1 + minus_freq_result) - 1
                time_result, freq_shift1 = time_freq(alice, minus_bob)
                new_bob = minus_bob / (1 + freq_shift1)
                plus = True
                continue
            except ValueError:
                always_plus = True
    
    time_result, freq_shift1 = time_freq(alice, bob) # do another pass to double check
    if abs(freq_shift1) <= abs(freq_shift2):
        freq_result = (1 + freq_shift1) * (1 + freq_result) - 1
        while abs(freq_shift1) > 1e-10:
            bob /= (1 + freq_shift1)
            time_result, freq_shift1 = time_freq(alice, bob)
            freq_result = (1 + freq_shift1) * (1 + freq_result) - 1
    else: # maybe need to redesign here
        backup_bob /= (1 + UPPER_LIMIT)
        freq_result = UPPER_LIMIT
        return pfind(alice, backup_bob)
    # print(f"{int(time_result):d}\t{int(freq_result * 1e12):d}\n")
    print(f"{int(time_result):d}\t{int(-freq_result / (1 + freq_result) * 2e34):d}\n")
    return time_result, freq_result

# time result: units of 1ns
# frequency result: units of 1e-12

import pathlib

def get_timestamp(dir_name, file_type, first_epoch, num_of_epochs, sep):
    epoch_dir = pathlib.Path(dir_name)
    timestamp = np.array([], dtype=np.float64)
    for i in range(num_of_epochs):
        epoch_name = epoch_dir / int2epoch(epoch2int(first_epoch) + i)
        reader = read_T1 if file_type == "T1" else read_T2
        timestamp = np.append(timestamp, reader(epoch_name)[0])  # TODO
        
    for i in range(num_of_epochs):
        epoch_name = epoch_dir / int2epoch(epoch2int(first_epoch) + sep + i)
        reader = read_T1 if file_type == "T1" else read_T2
        timestamp = np.append(timestamp, reader(epoch_name)[0])  # TODO
    return timestamp

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "")
    parser.add_argument("-d", help="SENDFILES")
    parser.add_argument("-D", help="T1FILES")
    parser.add_argument("-V", type = int, help = "verbosity")
    parser.add_argument("-e", help="first overlapping epoch between the two remotes")
    parser.add_argument("-n", type = int, default = 1, help = "number of epochs")
    parser.add_argument("-s", type = int, default = 6, help = "number of separation epochs")
    parser.add_argument("-q", type = int, default = 20,  help = "FFT buffer order, N = 2**q")
    parser.add_argument("-r", type = int, default = 1, help="desired timing resolution")
    parser.add_argument("-S", type = float, default = 6, help = "statistical significance threshold")


    if len(sys.argv) > 1:
        args = parser.parse_args()

        first_epoch = args.e
        num_of_epochs = args.n

        # alice: low count side - chopper - HeadT2 - sendfiles
        # bob: high count side - chopper2 - HeadT1 - t1files

        alice = get_timestamp(args.d, 'T2', args.e, args.n, args.s)
        bob = get_timestamp(args.D, 'T1', args.e, args.n, args.s)

        delta_tmax = args.r
        N = 2 ** args.q
        S_th = args.S

        pfind(alice, bob)