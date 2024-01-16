import warnings

import numpy as np
import pytest

# Automatically skip 80-bit precision checks if not supported
# Note: np.double only goes up to 1.79e308, while np.longdouble
#       supports up to 1.19e4932 (if implemented as 80-bit float).
#       For more information, see 'np.finfo(np.longdouble)'.
is_extfloat_supported = (np.longdouble("1e309") != np.inf)
if not is_extfloat_supported:
    warnings.warn("Extended precision float (80-bit) is not supported.")

def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )
    parser.addoption(
        "--float80", action="store_true", help="Check support for 80-bit extended precision floats"
    )
    parser.addoption(
        "--float64", action="store_true", help="Check support for 64-bit precision floats"
    )

def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")
    config.addinivalue_line("markers", "float80: mark test as requiring 80-bit float support")
    config.addinivalue_line("markers", "float64: mark test as requiring 64-bit float as fallback for extended precision container")

def pytest_collection_modifyitems(config, items):
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    if not config.getoption("--runslow"):
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)

    # Ensure float64 and float80 not given at the same time
    if config.getoption("--float80") and config.getoption("--float64"):
        raise ValueError("'--float64' and '--float80' cannot be supplied at the same time")
    
    skip_float80 = pytest.mark.skip(reason="no support for extended precision")
    skip_float64 = pytest.mark.skip(reason="extended precision supported, ignoring float64 tests")
    if ((not is_extfloat_supported) and (not config.getoption("--float80"))) \
            or config.getoption("--float64"):
        for item in items:
            if "float80" in item.keywords:
                item.add_marker(skip_float80)
    else:
        for item in items:
            if "float64" in item.keywords:
                item.add_marker(skip_float64)
