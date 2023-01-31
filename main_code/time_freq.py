import argparse
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.fft import fft, ifft
import sys

UPPER_LIMIT = 5e-5 # whether to put as an option
TIMESTAMP_RESOLUTION = 256
S_th = 6 #significance limit
Ta = 2**29 #acquisition time interval/ num of periods/ epochs
Ts = 6 * Ta #separation time interval
delta_tmax = 1 # the maximum acceptable delta_t to start tracking/ timing resolution
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

# Update:
# 1. use only the real part from the cross correlation to calculate find_max and statistical_significance
# 2. shall we calculate sigma using the same method as pfind.c or just this statistical significance?

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

def process_timestamp1(arr, start_time, delta_t):
    new_arr = arr[np.where((arr > start_time) & (arr <= start_time + Ta))]
    new_arr -= start_time
    bin_arr = np.zeros(N, dtype = np.int32)
    #t/delta_t mod N
    for i in range(np.size(new_arr)):
        bin_arr[math.floor(new_arr[i] // delta_t) % N] = 1
    return bin_arr

def process_timestamp(arr, start_time, delta_t):
    new_arr = arr[np.where((arr > start_time) & (arr <= start_time + Ta))]
    bin_arr = np.zeros(N, dtype = np.int32)
    #t/delta_t mod N
    for i in range(np.size(new_arr)):
        bin_arr[math.floor(new_arr[i] // delta_t) % N] = 1
    return bin_arr

def time_freq(arr_a, arr_b):
    delta_t = 256 #smallest discretization time, need to make this initial dt a free variable

    #generate arrays according to equation (6)
    T_start = max(arr_a[0], arr_b[0])

    bin_arr_a1 = process_timestamp1(arr_a, T_start, delta_t)
    bin_arr_b1 = process_timestamp1(arr_b, T_start, delta_t)
    arr_c1 = cross_corr(bin_arr_a1, bin_arr_b1)
    
    while statistical_significance(arr_c1) <= S_th:
        if delta_t > 5e6: # if delta_t goes too large and still cannot find the peak, means need to change the hyperparameters
            print("Cannot successfully find the peak!", file = sys.stderr)
            raise ValueError
        #halve the size of the array {ck} by adding entries pairwise
        arr_c1 = np.sum(arr_c1.reshape(-1, 2), axis = 1)
        delta_t *= 2 #doubles the effective time resolution

    #After this step, we can determine the peak position from the array and the effective time resolution
    delta_T1 = find_max(arr_c1)[0] * delta_t

    #now the delta_t is already the effective time resolution
    bin_arr_a2 = process_timestamp1(arr_a, T_start + Ts, delta_t)
    bin_arr_b2 = process_timestamp1(arr_b, T_start + Ts, delta_t)
    arr_c2 = cross_corr(bin_arr_a2, bin_arr_b2)
    delta_T2 = find_max(arr_c2)[0] * delta_t

    delta_u = (delta_T1 - delta_T2) / Ts

    if abs(delta_u) < 1e-10:
        while delta_t > delta_tmax:
            delta_t = delta_t / (Ts / Ta / math.sqrt(2))
            new_arr_b = arr_b - delta_T1
            new_arr_a1 = process_timestamp1(arr_a, T_start, delta_t)
            new_arr_b1 = process_timestamp1(new_arr_b, T_start, delta_t)
            new_arr_c1 = cross_corr(new_arr_a1, new_arr_b1)

            if statistical_significance(new_arr_c1) < 6: # if the peak is too low, high likely the result is wrong
                break

            delta_T1_correct = find_max(new_arr_c1)[0] * delta_t
            delta_T1 += delta_T1_correct
        
        print(delta_T1, 0, file = sys.stderr)
        return delta_T1, 0
 
    arr_a -= T_start
    arr_b -= T_start
    T_start = 0

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
    print(delta_T1, delta_u, file = sys.stderr)
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
        # need to set limit for plus & minus step size
        # if reach the limit still cannot obtain the correct result, means need to change the hyperparameters
    
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
    print(f"{int(time_result):d}\t{int(freq_result * 1e12):d}\n", file = sys.stdout)
    return time_result, freq_result

# time result: units of 1ns
# frequency result: units of 1e-12

def result(alice_timestamp, bob_timestamp): #remember to change legacy setting, default is False
    alice = read_a1(alice_timestamp)
    bob = read_a1(bob_timestamp)
    alice_time, alice_result = pfind(bob, alice) # bob as reference
    alice = read_a1(alice_timestamp)
    bob = read_a1(bob_timestamp)
    bob_time, bob_result = pfind(alice, bob) # alice as reference
    print((alice_result + 1) * (bob_result + 1), file = sys.stderr) # prove that frequency results converge
    print(alice_time * (1 + alice_result) + bob_time, file = sys.stderr) # prove that time offset results converge


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "")
    parser.add_argument("-d", help="RECEIVEFILES")
    parser.add_argument("-x", action="store_true", help="alice legacy format")
    parser.add_argument("-D", help="T1FILES")
    parser.add_argument("-X", action="store_true", help="bob legacy format") # may not be useful later but we just put here
    # parser.add_argument("-e", type = float, help="first overlapping epoch between the two remotes") # need to add this back when receive file from chopper
    parser.add_argument("-n", type = int, help = "number of epochs, Ta")
    parser.add_argument("-V", help = "verbosity")
    parser.add_argument("-q", type = int, help = "FFT buffer order, N")
    parser.add_argument("-r", type = int, help="desired timing resolution")


    if len(sys.argv) > 1:
        args = parser.parse_args()

        alice = read_a1(args.d, args.x)
        bob = read_a1(args.D, args.X)

        Ta = 2 ** args.n
        # T_start = args.e # maybe change in each iteration
        delta_tmax = args.r
        N = 2 ** args.q
        Ts = 6 * Ta

        pfind(alice, bob)

# python .\time_freq.py -d .\test_alice.dat -D .\test_bob.dat -n 29 -q 20 -r 1