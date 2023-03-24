import enum

class TSRES(enum.Enum):
    """Stores timestamp resolution information.
    
    Values assigned correspond to the number of units within a
    span of 1 nanosecond.
    """
    NS1 = 1
    PS125 = 8
    PS4 = 256
