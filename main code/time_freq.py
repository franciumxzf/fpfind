import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.fft import fft, ifft

UPPER_LIMIT = 5e-5
TIMESTAMP_RESOLUTION = 256
Ta = 2**29 #acquisition time interval
Ts = 6 * Ta #separation time interval
S_th = 6 #significance limit
delta_tmax = 1 # the maximum acceptable delta_t to start tracking
N = 2**20 #bin number, note that this N remains unchanged during the whole algorithm

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
    time_diff = 0
    max_value = arr[time_diff]
    for i in range(len(arr)):
        if arr[i] > max_value:
            time_diff = i
            max_value = arr[i]
    if time_diff > (np.size(arr) / 2):
        time_diff -= np.size(arr)
    return time_diff, max_value

def statistical_significance(arr):
    ck = np.max(arr)
    ck_average = arr.sum() / np.size(arr)
    S = (ck - ck_average) / np.std(arr)
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
            print("Cannot successfully find the peak!")
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
        
        print(delta_T1, 0)
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

        if statistical_significance(new_arr_c1) < 6: # if the peak is too low, high likely the result is wrong
            break

    delta_u = 1 / u - 1
    print(delta_T1, delta_u)
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
    print(time_result, freq_result)
    return time_result, freq_result

def result(alice_timestamp, bob_timestamp): #remember to change legacy setting, default is False
    alice = read_a1(alice_timestamp)
    bob = read_a1(bob_timestamp)
    alice_time, alice_result = pfind(bob, alice) # bob as reference
    alice = read_a1(alice_timestamp)
    bob = read_a1(bob_timestamp)
    bob_time, bob_result = pfind(alice, bob) # alice as reference
    print((alice_result + 1) * (bob_result + 1)) # prove that frequency results converge
    print(alice_time * (1 + alice_result) + bob_time) # prove that time offset results converge