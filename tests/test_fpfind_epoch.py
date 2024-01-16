import pytest

from fpfind.main import get_timestamp, fpfind

TESTCASES_FAST = [
    ('./data/epochs/20221109170747/sendfiles', './data/epochs/20221109170747/t1', 'ba2cfb33'),
    ('./data/epochs/20221109171401/sendfiles', './data/epochs/20221109171401/t1', 'ba2cfdeb'),
    # ('./data/epochs/20221109171525/sendfiles', './data/epochs/20221109171525/t1', 'ba2cfe88'),
    # ('./data/epochs/20221109171723/sendfiles', './data/epochs/20221109171723/t1', 'ba2cff63'),
    # ('./data/epochs/20221109171840/sendfiles', './data/epochs/20221109171840/t1', 'ba2cfff2'),
    # ('./data/epochs/dataset11/sendfiles', './data/epochs/dataset11/t1', 'ba2d38dc'),
    # ('./data/epochs/dataset16/sendfiles', './data/epochs/dataset16/t1', 'ba2d3b99'),
    # ('./data/epochs/dataset25/sendfiles', './data/epochs/dataset25/t1', 'ba2d4473'),
    # ('./data/epochs/dataset26/sendfiles', './data/epochs/dataset26/t1', 'ba2d45c3'),
    # ('./data/epochs/dataset27/sendfiles', './data/epochs/dataset27/t1', 'ba2d4714')
]

@pytest.mark.parametrize("SENDFILES_dir, T1_dir, first_epoch", TESTCASES_FAST)
def test_fpfind_epoch(SENDFILES_dir, T1_dir, first_epoch):
    num_of_epochs = 1
    sep = 6
    num_epochs_skipped = 0
    
    alice = get_timestamp(SENDFILES_dir, 'T2', first_epoch, num_epochs_skipped, num_of_epochs, sep)
    bob = get_timestamp(T1_dir, 'T1', first_epoch, num_epochs_skipped, num_of_epochs, sep)
    alice_copy = alice.copy()
    bob_copy = bob.copy()

    alice_time, alice_freq = fpfind(bob, alice)
    bob_time, bob_freq = fpfind(alice_copy, bob_copy)

    freq_invariance = (alice_freq + 1) * (bob_freq + 1)
    time_invariance = alice_time * (1 + alice_freq) + bob_time

    assert pytest.approx(freq_invariance, abs=1e-9) == 1  # to within 1ppb
    assert pytest.approx(time_invariance, abs=1) == 0  # to within +/- 1 ns

TESTCASES_SLOW = [
    ('./data/epochs/20221123165621/sendfiles', './data/epochs/20221123165621/t1', 'ba2d5734'),
    ('./data/epochs/dataset12/sendfiles', './data/epochs/dataset12/t1', 'ba2d392b'),
    ('./data/epochs/dataset13/sendfiles', './data/epochs/dataset13/t1', 'ba2d399a'),
    ('./data/epochs/dataset14/sendfiles', './data/epochs/dataset14/t1', 'ba2d3a18'),
    ('./data/epochs/dataset20/sendfiles', './data/epochs/dataset20/t1', 'ba2d42ec'),
    ('./data/epochs/dataset22/sendfiles', './data/epochs/dataset22/t1', 'ba2d43b1')
]

@pytest.mark.parametrize("SENDFILES_dir, T1_dir, first_epoch", TESTCASES_SLOW)
@pytest.mark.slow
def test_fpfind_epoch_slow(SENDFILES_dir, T1_dir, first_epoch):
    num_of_epochs = 4
    sep = 3
    num_epochs_skipped = 0
    
    alice = get_timestamp(SENDFILES_dir, 'T2', first_epoch, num_epochs_skipped, num_of_epochs, sep)
    bob = get_timestamp(T1_dir, 'T1', first_epoch, num_epochs_skipped, num_of_epochs, sep)
    alice_copy = alice.copy()
    bob_copy = bob.copy()

    alice_time, alice_freq = fpfind(bob, alice)
    bob_time, bob_freq = fpfind(alice_copy, bob_copy)

    freq_invariance = (alice_freq + 1) * (bob_freq + 1)
    time_invariance = alice_time * (1 + alice_freq) + bob_time

    assert pytest.approx(freq_invariance, abs=1e-9) == 1  # to within 1ppb
    assert pytest.approx(time_invariance, abs=1) == 0  # to within +/- 1 ns
