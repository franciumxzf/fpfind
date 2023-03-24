import numpy as np
import matplotlib.pyplot as plt
import parse_timestamps as parser

Ta = 2**27
N = 2**18
# binwidth = 32

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

# alice1 = parser.read_a1('./data/1_rawevents/raw_alice_20221109170747.dat', legacy = True)
# bob1 = parser.read_a1('./data/1_rawevents/raw_bob_20221109170747.dat', legacy = True)
# start = max(alice1[0], bob1[0])
# alice_pro = alice1[np.where((alice1 > start) & (alice1 <= start + Ta))]
# bob_pro = bob1[np.where((bob1 > start) & (bob1 <= start + Ta))]
# print(alice_pro.size)
# print(bob_pro.size)
# g2alicebob = g2(alice_pro, bob_pro)
# plt.plot(g2alicebob)
# plt.show()
