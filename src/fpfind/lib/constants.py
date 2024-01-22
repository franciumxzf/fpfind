import enum

class TSRES(enum.Enum):
    """Stores timestamp resolution information.

    Values assigned correspond to the number of units within a
    span of 1 nanosecond.
    """
    NS1 = 1    # S-Fifteen TDC1 timestamp (nominal 2ns resolution)
    PS125 = 8  # CQT Red timestamp
    PS4 = 256  # S-Fifteen TDC2 timestamp

EPOCH_LENGTH = 1 << 29  # from filespec
FCORR_AMAXBITS = -13  # from 'freqcd.c'

# Derived constants
MAX_FCORR = 2**-13
