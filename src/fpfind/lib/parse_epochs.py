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
import pathlib
from struct import pack, unpack
from typing import NamedTuple

import numpy as np

from fpfind import TSRES

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

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
    file_name = str(file_name)
    with open(file_name, 'rb') as f:
        head_info = f.read(4 * 5)
    headt1 = HeadT1._make(unpack('iIIii', head_info))
    if (headt1.tag != 0x101 and headt1.tag != 1) :
        logger.warning(f'{file_name} is not a Type2 header file')
    if hex(headt1.epoch) != ('0x' + file_name.split('/')[-1]):
        logger.warning(f'Epoch in header {headt1.epoch} does not match epoc filename {file_name}')
    return headt1

def read_T2_header(file_name: str):
    file_name = str(file_name)
    with open(file_name, 'rb') as f:
        head_info = f.read(4*6)
    headt2 = HeadT2._make(unpack('iIIiii', head_info))
    if (headt2.tag != 0x102 and headt2.tag != 2) :
        logger.warning(f'{file_name} is not a Type2 header file')
    if hex(headt2.epoch) != ('0x' + file_name.split('/')[-1]):
        logger.warning(f'Epoch in header {headt2.epoch} does not match epoc filename {file_name}')
    return headt2

def read_T3_header(file_name: str):
    file_name = str(file_name)
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
    total_seconds = (int(epoch, base=16) << 32) * 125e-12
    return dt.datetime.fromtimestamp(total_seconds)

def epoch2int(epoch):
    return int(epoch, base=16)

def int2epoch(value):
    return hex(value)[2:]


# Epoch readers
# Implemented as per filespec [1]

def read_T1(
        filename: str,
        full_epoch: bool = False,
        resolution: TSRES = TSRES.NS1,
        fractional: bool = True,
    ):
    """Returns timestamp and detector information from T1 files.
    
    Timestamp and detector information follow the formats specified in
    'parse_timestamps' module, for interoperability.

    By default, the function returns timestamps in units of 1ns with as
    high a resolution as possible (i.e. including fractional values).
    
    TODO

    'full' is False by default, which indicates only the raw-timestamp
    -equivalent information is returned, i.e. timestamp in units of 1ns
    stored as a 64-bit integer.
    'full' set to True returns full timestamps (timestamps + full epoch
    information) stored as 128-bit float.
    The reason for this difference is due to the full timestamp taking up
    more than 64-bits equivalent information in units of 1ns, so only a
    float can support this dynamic range.

    Note 128-bit float has precision of 113 bits, while 64-bit float only has
    53-bit precision (insufficient resolution).

    If 'raw' is True, timestamps are stored in units of 4ps instead of 1ns.
    
    Raises:
        ValueError: Number of epochs does not match length in header.
    """
    # Header details
    header = read_T1_header(filename)
    length = header.length_bits

    timestamps = []
    bases = []

    # For each timestamp event, 54-bits correspond to timestamp,
    # of which 17-bit MSB is LSB of epoch, 37-bit LSB is 125ps resolution timestamp
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

            # Timestamp in units of 4ps, stored in 54-bit MSB
            timestamp = value >> 10
            timestamps.append(timestamp)

            base = value & 0b1111
            bases.append(base)
    
    # Validity checks
    if length and len(timestamps) != length:
        raise ValueError("Number of epochs do not match length specified in header")
    
    # Add epoch if required, at most 32-17 = 15 bits
    epoch_msb = header.epoch >> 17
    
    # Right now in units of 4ps
    # Check if need to store floating point
    if fractional:
        # Convert to desired resolution
        timestamps = np.array(timestamps, dtype=np.float128)
        timestamps = timestamps / (TSRES.PS4.value/resolution.value) 
        epoch_msb = np.float128(epoch_msb << 54)
        epoch_msb = epoch_msb / (TSRES.PS4.value/resolution.value)

    # Python integer objects can be arbitrarily long
    elif resolution in (TSRES.PS4,) and full_epoch:
        timestamps = np.array(timestamps, dtype=object)
        epoch_msb = int(epoch_msb) << 54
    
    # Everything of resolution 125 ps and larger can fit
    # in 64-bit unsigned integer, i.e. PS125, NS1.
    else:
        bitdiff = round(np.log2(TSRES.PS4.value/resolution.value))
        timestamps = np.array(timestamps, dtype=np.uint64)
        timestamps = timestamps >> bitdiff
        epoch_msb = np.uint64(epoch_msb) << np.uint64(54-bitdiff)

    # Add epoch
    if full_epoch:
        timestamps += epoch_msb
    
    return timestamps, np.array(bases)


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


def read_T2(
        filename: str,
        full_epoch: bool = False,
        resolution: TSRES = TSRES.NS1,
        fractional: bool = True,
    ):
    """Returns timestamp and detector information from T2 files.
    
    Timestamp and detector information follow the formats specified in
    'parse_timestamps' module, for interoperability.

    Raises:
        ValueError: Length and termination bits inconsistent with filespec.
    """
    if resolution not in TSRES:
        raise ValueError(
            "Timestamp resolution must be one of enumeration TSRES"
        )

    # Header details
    header = read_T2_header(filename)
    timeorder = header.timeorder
    timeorder_extended = 32  # just being pedantic
    basebits = header.base_bits
    length = header.length_bits

    # Accumulators
    timestamps = []  # all timestamps, units of 125ps
    bases = []

    # Add 17-bit epoch LSB as per regular timestamp spec,
    # i.e. (17-bit epoch LSB)(32-bit timestamp) => 49-bits
    epoch_lsb = header.epoch & ((1 << 17) - 1)
    timestamp = np.uint64(epoch_lsb << 32)  # current timestamp, deltas stored in T2
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
    
    # Add full epoch if required
    epoch_msb = header.epoch >> 17
    
    # Right now in units of 125ps
    # Check if need to store floating point
    if fractional:
        # Convert to desired resolution
        timestamps = np.array(timestamps, dtype=np.float128)
        timestamps = timestamps / (TSRES.PS125.value/resolution.value) 
        epoch_msb = np.float128(epoch_msb << 49)
        epoch_msb = epoch_msb / (TSRES.PS125.value/resolution.value)

    # Python integer objects can be arbitrarily long
    # for resolutions smaller than 125ps, i.e. larger value
    elif resolution.value > TSRES.PS125.value:
        bitdiff = round(np.log2(resolution.value/TSRES.PS125.value))
        timestamps = np.array(timestamps, dtype=object)
        timestamps = timestamps << bitdiff
        epoch_msb = int(epoch_msb) << (49+bitdiff)
    
    # Everything of resolution 125 ps and larger can fit
    # in 64-bit unsigned integer, i.e. PS125, NS1.
    else:
        bitdiff = round(np.log2(TSRES.PS125.value/resolution.value))
        timestamps = np.array(timestamps, dtype=np.uint64)
        timestamps = timestamps >> bitdiff
        epoch_msb = np.uint64(epoch_msb) << np.uint64(49-bitdiff)

    # Add epoch
    if full_epoch:
        timestamps += epoch_msb

    return timestamps, np.array(bases)


def write_T1(directory, full_epoch, timestamps, detectors):
    # Fit epoch to within 32-bits LSB
    full_epoch &= ((1<<32)-1)

    directory = pathlib.Path(directory)
    target = directory / f"{full_epoch:x}"
    
    # Broadcast detectors if single value given
    if np.array(detectors).ndim == 0:
        detectors = np.ones(len(timestamps)) * detectors
    elif len(detectors) != len(timestamps):
        raise ValueError("Lengths of timestamps and detectors must match")
    
    with open(target, "wb") as f:
        header = pack("iIIii", 0x101, full_epoch, len(timestamps), 49, 4)
        f.write(header)

        # Write events, 4ps resolution
        # 17-bit LSB epoch fits into a timestamp event
        timestamp_epoch = ((full_epoch & ((1<<17)-1)) << (32 + 5))
        for timestamp, detector in zip(timestamps, detectors):
            full_timestamp = timestamp_epoch + timestamp
            event = (full_timestamp << 10) + int(detector)

            # Write high word, then low word
            f.write(pack("<II", event >> 32, event & 0xffff_ffff))
        
        # Termination
        f.write(pack("II", 0, 0))

    return target
