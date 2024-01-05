#!/usr/bin/env python3
import argparse
import numpy as np
import logging
import math
from scipy.fft import fft, ifft
import sys
import pathlib

from fpfind.lib.parse_epochs import read_T1, read_T2, epoch2int, int2epoch

DELTA_U_STEP = 1e-5 # whether to put as an option
DELTA_U_MAX = 0
TIMESTAMP_RESOLUTION = 256
EPOCH_LENGTH = 2 ** 29
S_th = 6 #significance limit
Ta = 2**29 #acquisition time interval/ num of periods/ epochs
Ts = 6 * Ta #separation time interval
delta_tmax = 1 # the maximum acceptable delta_t to start tracking/ timing resolution
N = 2**20 #bin number, note that this N remains unchanged during the whole algorithm/ buffer order

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
    S = (ck - ck_average) / np.std(arr)
    return S

def process_timestamp(arr, start_time, delta_t):
    new_arr = arr[np.where((arr > start_time) & (arr <= start_time + Ta))]
    bin_arr = np.zeros(N, dtype = np.int32)
    #t/delta_t mod N
    result = np.int32((new_arr // delta_t) % N)
    bin_arr[result] = 1
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
    # print("time offset: %d, frequency offset: %f", delta_T1, delta_u)
    # logging.debug("time offset: %d, frequency offset: %f", delta_T1, delta_u)
    return delta_T1, delta_u

def fpfind(alice, bob):
    iterating_list = np.arange(1, int(DELTA_U_MAX // DELTA_U_STEP + 1) * 2) // 2
    iterating_list[::2] *= -1
    # print(iterating_list * DELTA_U_STEP)
    for i in iterating_list:
        # print("Testing:", DELTA_U_STEP * i)
        try:
            delta_T, delta_u = time_freq(alice, bob / (1 + DELTA_U_STEP * i))
        except:
            continue
        # print(delta_T, delta_u)
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
    parser.add_argument("--skip", type = int, default = 0, help = "number of skip epochs")

    if len(sys.argv) > 1:
        args = parser.parse_args()

        first_epoch = args.e
        skip_epoch = args.skip
        num_of_epochs = args.n

        # alice: low count side - chopper - HeadT2 - sendfiles
        # bob: high count side - chopper2 - HeadT1 - t1files

        alice = get_timestamp(args.d, 'T2', first_epoch, skip_epoch, num_of_epochs, args.s)
        bob = get_timestamp(args.D, 'T1', first_epoch, skip_epoch, num_of_epochs, args.s)

        delta_tmax = args.r
        Ta = num_of_epochs * EPOCH_LENGTH
        Ts = args.s * Ta
        N = 2 ** args.q
        S_th = args.S

        td, fd = fpfind(bob, alice)
        print(fd)
        # print(f"{round(td):d}\t{round(fd * (1 << 34)):d}\n")
