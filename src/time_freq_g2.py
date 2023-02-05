import parse_timestamps as parser
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
from sklearn.linear_model import LinearRegression

delta_umax = 1 #maximum expected frequency
delta_Tmax = 1 #maximum expected time difference
r1 = 1 #background rate of timestamp 1
r2 = 1 #background rate of timestamp 2
rs = 1 #signal rate of true coincidences

Ta = 2**29 #acquisition time interval
Ts = 10 * Ta #separation time interval
#T_start = 20533409000000 #start point
#timestamp_resolution = 125e-12
S_th = 6 #significance limit
delta_tmax = 1 # the maximum acceptable delta_t to start tracking
N = 2**20 #bin number, note that this N remains unchanged during the whole algorithm
#delta_t = 2048

def find_max(arr):
    time_diff = 0
    max_value = arr[time_diff]
    for i in range(len(arr)):
        if arr[i] > max_value:
            time_diff = i
            max_value = arr[i]
    # if time_diff > (N//2):
    #     time_diff -= N
	#print(time_diff)
	#print(max_value)
    return time_diff, max_value

def find_max1(arr):
    time_diff = 0
    max_value = arr[time_diff]
    for i in range(len(arr)):
        if arr[i] > max_value:
            time_diff = i
            max_value = arr[i]
    if time_diff > (N//2):
        time_diff -= N
	#print(time_diff)
	#print(max_value)
    return time_diff, max_value

def statistical_significance(arr):
    ck = find_max(arr)[1]
    ck_average = arr.sum() / np.size(arr)
    S = (ck - ck_average) / np.std(arr)
    return S

def g2(arr1, arr2, start_time, delta_t):
    maxdelay = 10000000
    arr1 = arr1[np.where((arr1 > start_time) & (arr1 <= start_time + Ta))]
    arr2 = arr2[np.where((arr2 > start_time) & (arr2 <= start_time + Ta))]
    arr3 = np.zeros(N, dtype = np.int32)
    delay = 0

    for event in arr2:
        while arr1.size > 1:
            if (event - arr1[0] >= maxdelay):
                # np.delete(arr1, 0)
                arr1 = arr1[1:]
                # del arr1[:1]
            else: break
        for stamp in arr1:
            delay = event - stamp
            arr3[int(delay // delta_t) % N] += 1

    return arr3

def time_freq_g2(arr_a, arr_b):
    delta_t = 32 #smallest discretization time

    T_start = max(arr_a[10], arr_b[10])
    arr_c1 = g2(arr_a, arr_b, T_start, delta_t)
    
    while statistical_significance(arr_c1) <= S_th:
        #halve the size of the array {ck} by adding entries pairwise
        plt.plot(arr_c1)
        plt.show()
        print(statistical_significance(arr_c1))
        print(delta_t)
        arr_c1 = np.sum(arr_c1.reshape(-1, 2), axis = 1)
        delta_t *= 2 #doubles the effective time resolution
    
    plt.plot(arr_c1)
    plt.show()
    #print(delta_t)
    #After this step, we can determine the peak position from the array and the effective time resolution
    delta_T1 = find_max(arr_c1)[0] * delta_t

    arr_c2 = g2(arr_a, arr_b, T_start + Ts, delta_t)
    delta_T2 = find_max(arr_c2)[0] * delta_t
    plt.plot(arr_c2)
    plt.show()
    #print(delta_T2)

    delta_u = (delta_T1 - delta_T2) / Ts

    if abs(delta_u) < 1e-10:
        while delta_t > delta_tmax:
            delta_t = delta_t / (Ts / Ta / math.sqrt(2))
            new_arr_b = arr_b - delta_T1
            new_arr_c1 = g2(arr_a, new_arr_b, T_start, delta_t)
            #plt.plot(new_arr_c1)
            #plt.show()
            delta_T1_correct = find_max1(new_arr_c1)[0] * delta_t

            delta_T1 += delta_T1_correct
        
        return delta_T1, 0

    arr_a -= T_start
    arr_b -= T_start
    T_start = 0

    new_arr_b = (arr_b - delta_T1) / (1 - delta_u)
    delta_T1_correct = 0
    delta_u_correct = 0
    u = 1 / (1 - delta_u)
    
    while delta_t > delta_tmax:
        # print(delta_u, delta_t, delta_T1, delta_T2, delta_T1_correct)
        print(f"du': {delta_u_correct}, dT': {delta_T1_correct}, dt: {delta_t}")
        delta_t = delta_t / (Ts / Ta / math.sqrt(2))
        new_arr_b = (new_arr_b  - delta_T1_correct) / (1 - delta_u_correct)

        new_arr_c1 = g2(arr_a, new_arr_b, T_start, delta_t)
        plt.plot(new_arr_c1)
        plt.show()
        delta_T1_correct = find_max1(new_arr_c1)[0] * delta_t

        new_arr_c2 = g2(arr_a, new_arr_b, T_start + Ts, delta_t)
        plt.plot(new_arr_c2)
        plt.show()
        delta_T2_correct = find_max1(new_arr_c2)[0] * delta_t

        delta_T1 = delta_T1_correct + (delta_T1) / (1 - delta_u_correct)
        delta_T2 = delta_T2_correct + delta_T2 / (1 - delta_u_correct)
        delta_u_correct = (delta_T1_correct - delta_T2_correct) / Ts
        
        u *= (1 / (1 - delta_u_correct))
    
    # print(delta_T1)
    #print(delta_T2)
    #print(1 / u - 1)

    delta_u = 1 / u - 1
    
    return delta_T1, delta_u