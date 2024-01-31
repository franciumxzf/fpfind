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
NTP_MAXDELAY_NS = 100e6  # 100ms in very asymmetric channels

# Derived constants
MAX_FCORR = 2**FCORR_AMAXBITS

class PeakFindingFailed(ValueError):
    def __init__(
            self,
            message,
            significance=None,
            resolution=None,
            dt1=None,
            dt2=None,
            dt=None,
            df=None,
        ):
        self.message = message
        self.s = significance
        self.r = resolution
        self.dt1 = dt1
        self.dt2 = dt2
        self.dt = dt
        self.df = df
        super().__init__(message)

    def __str__(self):
        text = self.message
        suppl = []
        if self.s is not None:
            suppl.append(f"S={self.s:8.3f}")
        if self.r is not None:
            suppl.append(f"r={self.r:5.0f}")
        if self.dt1 is not None:
            suppl.append(f"dt1={self.dt1:11.0f}")
        if self.dt2 is not None:
            suppl.append(f"dt2={self.dt2:11.0f}")
        if self.dt is not None:
            suppl.append(f"dt={self.dt:11.0f}")
        if self.df is not None:
            suppl.append(f"df={self.df*1e6:.4f}ppm")
        if suppl:
            text = f"{text} ({', '.join(suppl)})"
        return text
