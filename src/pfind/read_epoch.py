import numpy as np
import os
from pathlib import Path
from struct import unpack
import sys
from typing import NamedTuple

# sendfiles - chopper - HeadT2 - low count side
# t1files - chopper2 - HeadT1 - high count side
# for t1 type, they are actually just the raw timestamp defination

TIMESTAMP_RESOLUTION = 256

class HeadT1(NamedTuple):
    tag: int
    epoch: int
    length_bits: int
    bits_per_entry: int
    base_bits: int

class HeadT2(NamedTuple):
    tag: int
    epoch: int
    length_bits: int
    timeorder: int
    base_bits: int
    protocol: int

def read_T1_header(file_name: str):
    with open(file_name, 'rb') as f:
        head_info = f.read(4 * 5)
    headt1 = HeadT1._make(unpack('iIIii', head_info))
    if (headt1.tag != 0x101 and headt1.tag != 1) :
        print(f'{file_name} is not a Type2 header file', file = sys.stderr)
    if hex(headt1.epoch) != ('0x' + file_name.split('/')[-1]):
        print(f'Epoch in header {headt1.epoch} does not match epoc filename {file_name}', file = sys.stderr)    
    return headt1

def read_T2_header(file_name: str):
    with open(file_name, 'rb') as f:
        head_info = f.read(4*6)
    headt2 = HeadT2._make(unpack('iIIiii', head_info))
    if (headt2.tag != 0x102 and headt2.tag != 2) :
        print(f'{file_name} is not a Type2 header file', file = sys.stderr)
    if hex(headt2.epoch) != ('0x' + file_name.split('/')[-1]):
        print(f'Epoch in header {headt2.epoch} does not match epoc filename {file_name}', file = sys.stderr)
    return headt2

def read_T2_epochs(dir_name: str): # for T2 type
    epochs = os.listdir(dir_name)
    event_timings = []
    for i in range(len(epochs)):
        epoch_name = dir_name + '/' + epochs[i]
        with open(epoch_name, 'rb') as f:
            header = read_T2_header(epoch_name)
            f.seek(4*6)

            num_of_events = header.length_bits
            basebits = header.base_bits
            timeorder = header.timeorder
            timing = header.epoch

            events_count = 0
            extend = False
            # entry = basebits + timeorder

            while events_count != num_of_events:
                event = int.from_bytes(f.read(4), byteorder = 'little')
                if extend:
                    timing += event
                    event_timings.append(timing / TIMESTAMP_RESOLUTION)
                    events_count += 1
                    extend = False
                    continue

                event_timeorder = event & ((1 << timeorder) - 1)
                # event_basebits = (event >> timeorder) & ((1 << basebits) - 1)

                if event_timeorder == 0:
                    extend = True
                    continue
                    
                timing += event_timeorder
                event_timings.append(timing / TIMESTAMP_RESOLUTION)
                events_count += 1
    
    return event_timings

def read_T1_epochs(dir_name: str): # for T1 type
    epochs = os.listdir(dir_name)

    for i in range(len(epochs)):
        epoch_name = dir_name + '/' + epochs[i]
        with open(epoch_name, 'rb') as f:
            data = np.fromfile(file=f, dtype="=I").reshape(-1, 2)
        t = ((np.uint64(data[:, 1]) << 22) + (data[:, 0] >> 10)) / TIMESTAMP_RESOLUTION
    return t