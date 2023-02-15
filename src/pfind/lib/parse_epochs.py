#!/usr/bin/env python3
# Justin, 2023-02-15
"""Takes in an epoch and parses it into timestamps.

Currently only supports T1 and T2 epoch unpacking.

References:
    [1] File specification: https://github.com/s-fifteen-instruments/qcrypto/blob/Documentation/docs/source/file%20specification.rst
    [2] Epoch header definitions: https://github.com/s-fifteen-instruments/QKDServer/blob/master/S15qkd/utils.py
"""

import datetime as dt
import logging
from struct import unpack
from typing import NamedTuple

import numpy as np

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

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

class HeadT3(NamedTuple):
    tag: int
    epoch: int
    length_entry: int
    bits_per_entry: int


# Header readers
# Extracted from ref [2]

def read_T1_header(file_name: str):
    with open(file_name, 'rb') as f:
        head_info = f.read(4 * 5)
    headt1 = HeadT1._make(unpack('iIIii', head_info))
    if (headt1.tag != 0x101 and headt1.tag != 1) :
        logger.warning(f'{file_name} is not a Type2 header file')
    if hex(headt1.epoch) != ('0x' + file_name.split('/')[-1]):
        logger.warning(f'Epoch in header {headt1.epoch} does not match epoc filename {file_name}')
    return headt1

def read_T2_header(file_name: str):
    with open(file_name, 'rb') as f:
        head_info = f.read(4*6)
    headt2 = HeadT2._make(unpack('iIIiii', head_info))
    if (headt2.tag != 0x102 and headt2.tag != 2) :
        logger.warning(f'{file_name} is not a Type2 header file')
    if hex(headt2.epoch) != ('0x' + file_name.split('/')[-1]):
        logger.warning(f'Epoch in header {headt2.epoch} does not match epoc filename {file_name}')
    return headt2

def read_T3_header(file_name: str):
    with open(file_name, 'rb') as f:
        head_info = f.read(4*4)
    headt3 = HeadT3._make(unpack('iIIi', head_info))
    if (headt3.tag != 0x103 and headt3.tag != 3) :
        logger.warning(f'{file_name} is not a Type3 header file')
    if hex(headt3.epoch) != ('0x' + file_name.split('/')[-1]):
        logger.warning(f'Epoch in header {headt3.epoch} does not match epoc filename {file_name}')
    return headt3


# Epoch utility functions

def date2epoch(datetime=None):
    """Returns current epoch number as hex.
    
    Epochs are in units of 2^32 * 125e-12 == 0.53687 seconds.

    Args:
        datetime: Optional datetime to convert to epoch.
    """
    if datetime is None:
        datetime = dt.datetime.now()
    
    total_seconds = int(datetime.timestamp())
    epoch_val = int(total_seconds/125e-12) >> 32
    return f"{epoch_val:08x}"


def epoch2date(epoch):
    """Returns datetime object corresponding to epoch.
    
    Epoch must be in hex format, e.g. "ba1f36c0".

    Args:
        epoch: Epoch in hex format, e.g. "ba1f36c0".
    """
    total_seconds = int(epoch, base=16) << 32 * 125e-12
    return dt.datetime.fromtimestamp(total_seconds)


# Epoch readers
# Implemented as per filespec [1]

def read_T1(filename: str):
    """Returns timestamp and detector information from T1 files.
    
    Timestamp and detector information follow the formats specified in
    'parse_timestamps' module, for interoperability.

    Raises:
        ValueError: Number of epochs does not match length in header.
    """
    # Header details
    header = read_T1_header(filename)
    length = header.length_bits

    timestamps = []
    bases = []
    with open(filename, "rb") as f:
        f.read(5 * 4)  # dump header
        while True:
            event_high = f.read(4)
            event_low = f.read(4)

            value = \
                (int.from_bytes(event_high, byteorder="little") << 32) \
                + int.from_bytes(event_low, byteorder="little")
            
            if value == 0:
                break  # termination

            timestamp = (value >> 10) / TIMESTAMP_RESOLUTION
            timestamps.append(timestamp)

            base = value & 0b1111
            bases.append(base)
    
    # Validity checks
    if length and len(timestamps) != length:
        raise ValueError("Number of epochs do not match length specified in header")

    return np.array(timestamps), np.array(bases)


def extract_bits(msb_size: int, buffer: int, size: int, fileobject=None):
    """Return 'msb_size'-bits from MSB of buffer, and the rest.
    
    If insufficient bits to load value, data is automatically loaded
    from file 'fileobject' provided (must be already open in "rb" mode).

    Raises:
        ValueError: msb_size > size but fileobject not provided.
    
    Example:

        >>> msb_size = 4
        >>> buffer = 0b0101110; size = 7
        >>> extract_bits(msb_size, buffer, size)
        (5, 6, 3)

        Explanation:
          - msb => 0b0101 = 4
          - buffer => 0b110 = 5

    Note:
        Integrating with 'append_word_from_file' since this
        operation always precedes the value retrieval. This function
        can be reused in bit-packing schemes of other epoch types.
    """
    # Append word from fileobject if insufficient bits
    if size < msb_size and fileobject:
        buffer <<= 32
        buffer += int.from_bytes(fileobject.read(4), byteorder="little")
        size += 32
    
    # Extract MSB from buffer
    msb = buffer >> (size - msb_size)
    buffer &= (1 << (size - msb_size)) - 1
    size -= msb_size
    logging.debug("Extracted %d bytes: %d", msb_size, msb)
    return msb, buffer, size


def read_T2(filename: str):
    """Returns timestamp and detector information from T2 files.
    
    Timestamp and detector information follow the formats specified in
    'parse_timestamps' module, for interoperability.

    Raises:
        ValueError: Length and termination bits inconsistent with filespec.
    """
    # Header details
    header = read_T2_header(filename)
    timeorder = header.timeorder
    timeorder_extended = 32  # just being pedantic
    basebits = header.base_bits
    length = header.length_bits

    # Accumulators
    timestamps = []  # all timestamps, units of 125ps
    bases = []
    timestamp = 0  # current timestamp, deltas stored in T2
    buffer = 0
    size = 0  # number of bits in buffer
    with open(filename, "rb") as f:
        f.read(24)  # remove header
        while True:

            # Read timing information
            logging.debug("Extracting timing...")
            timing, buffer, size = \
                extract_bits(timeorder, buffer, size, f)

            # End-of-file check
            if timing == 1:
                base, buffer, size = extract_bits(basebits, buffer, size, f)
                if base != 0:
                    raise ValueError(
                        "File inconsistent with T2 epoch definition: "
                        "Terminal base bits not zero"
                    )
                break

            # Check if timing is actually in extended format, i.e. 32-bits
            if timing == 0:
                logging.debug("Extracting extended timing...")
                timing, buffer, size = \
                    extract_bits(timeorder_extended, buffer, size, f)

            timestamp += timing
            timestamps.append(timestamp)

            # Extract detector pattern, but not important here
            # Note we are guaranteed (timeorder+basebits < 32)
            logging.debug("Extracting base...")
            base, buffer, size = extract_bits(basebits, buffer, size, f)
            bases.append(base)
    
    # Validity checks
    if length and len(timestamps) != length:
        raise ValueError("Number of epochs do not match length specified in header")
    
    # Convert to units of ns
    timestamps = np.array(timestamps, dtype=np.float64)
    timestamps *= (32/TIMESTAMP_RESOLUTION)
    return timestamps, np.array(bases)
    