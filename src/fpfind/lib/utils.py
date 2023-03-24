def get_overlap(*arrays):
    """Returns right-truncated arrays of largest possible common length.
    
    Used for comparing timestamps of different length, e.g. raw timestamps
    vs chopper-generated individual epochs.
    """
    overlap = min(map(len, arrays))
    arrays = [a[:overlap] for a in arrays]
    return overlap, arrays
