import pytest

from fpfind.main import fpfind
from fpfind.lib.parse_timestamps import read_a1 as _read_a1

# Extract only timestamp data
def read_a1(*args, **kwargs):
    t, p = _read_a1(*args, **kwargs)
    return t

TESTCASES_FAST = [
    # Ta = 2**29, Ts = 6 * Ta, N = 2**20
    ('./data/raw_alice_bob/raw_alice_20221109170747.dat', './data/raw_alice_bob/raw_bob_20221109170747.dat'),
    ('./data/raw_alice_bob/raw_alice_20221109171401.dat', './data/raw_alice_bob/raw_bob_20221109171401.dat'),
    # ('./data/raw_alice_bob/raw_alice_20221109171525.dat', './data/raw_alice_bob/raw_bob_20221109171525.dat'),
    # ('./data/raw_alice_bob/raw_alice_20221109171723.dat', './data/raw_alice_bob/raw_bob_20221109171723.dat'),
    # ('./data/raw_alice_bob/raw_alice_20221109171840.dat', './data/raw_alice_bob/raw_bob_20221109171840.dat'),
    # ('./data/datasets/20221212_dataset11_0Hz_4kpps_e18_ch1.Aa1.dat', './data/datasets/20221212_dataset11_1200Hz_4kpps_e18_ch4.Aa1.dat'),
    # ('./data/datasets/20221212_dataset16_xHz_4kpps_e18_ch1.Aa1.dat', './data/datasets/20221212_dataset16_xHz_4kpps_e18_ch4.Aa1.dat'),
    # ('./data/datasets/20221212_dataset25_xHz_47kpps_e14_ch1.Aa1.dat', './data/datasets/20221212_dataset25_xHz_47kpps_e14_ch4.Aa1.dat'),
    # ('./data/datasets/20221212_dataset26_xHz_28kpps_e16_ch1.Aa1.dat', './data/datasets/20221212_dataset26_xHz_28kpps_e16_ch4.Aa1.dat'),
    # ('./data/datasets/20221212_dataset27_xHz_10kpps_e17_ch1.Aa1.dat', './data/datasets/20221212_dataset27_xHz_10kpps_e17_ch4.Aa1.dat'),
]

@pytest.mark.parametrize("alice, bob", TESTCASES_FAST)
#@pytest.mark.slow
def test_fpfind(alice, bob):
    alice1 = read_a1(alice, True)
    bob1 = read_a1(bob, True)
    alice_time, alice_result = fpfind(bob1, alice1)

    alice1 = read_a1(alice, True)
    bob1 = read_a1(bob, True)
    bob_time, bob_result = fpfind(alice1, bob1)
    
    freq_invariance = (alice_result + 1) * (bob_result + 1)
    time_invariance = alice_time * (1 + alice_result) + bob_time

    assert pytest.approx(freq_invariance, abs=1e-9) == 1  # to within 1ppb
    assert pytest.approx(time_invariance, abs=1) == 0  # to within +/- 1 ns

TESTCASES_SLOW = [
    # Ta = 2**31, Ts = 6 * Ta, N = 2**23
    ('./data/raw_alice_bob/raw_alice_20221123165621.dat', './data/raw_alice_bob/raw_bob_20221123165621.dat'),
    ('./data/datasets/20221212_dataset12_0Hz_4kpps_e18_ch1.Aa1.dat', './data/datasets/20221212_dataset12_1500Hz_4kpps_e18_ch4.Aa1.dat'),
    ('./data/datasets/20221212_dataset13_0Hz_4kpps_e18_ch1.Aa1.dat', './data/datasets/20221212_dataset13_2000Hz_4kpps_e18_ch4.Aa1.dat'),
    ('./data/datasets/20221212_dataset14_xHz_4kpps_e18_ch1.Aa1.dat', './data/datasets/20221212_dataset14_xHz_4kpps_e18_ch4.Aa1.dat'),
    ('./data/datasets/20221212_dataset20_0Hz_47kpps_e14_ch1.Aa1.dat', './data/datasets/20221212_dataset20_1200Hz_47kpps_e14_ch4.Aa1.dat'),
    ('./data/datasets/20221212_dataset22_0Hz_47kpps_e14_ch1.Aa1.dat', './data/datasets/20221212_dataset22_2000Hz_47kpps_e14_ch4.Aa1.dat')
]
@pytest.mark.parametrize("alice, bob", TESTCASES_SLOW)
@pytest.mark.slow
def test_fpfind_slow(alice, bob):
    alice1 = read_a1(alice, True)
    bob1 = read_a1(bob, True)
    alice_time, alice_result = fpfind(bob1, alice1)

    alice1 = read_a1(alice, True)
    bob1 = read_a1(bob, True)
    bob_time, bob_result = fpfind(alice1, bob1)
    
    freq_invariance = (alice_result + 1) * (bob_result + 1)
    time_invariance = alice_time * (1 + alice_result) + bob_time

    assert pytest.approx(freq_invariance, abs=1e-9) == 1  # to within 1ppb
    assert pytest.approx(time_invariance, abs=1) == 0  # to within +/- 1 ns