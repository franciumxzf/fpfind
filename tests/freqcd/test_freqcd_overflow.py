# TODO: Formalize testcases

"""TESTCASE 1: Check edge cases.

# Note: Max raw timestamp value is 2**54 == 18014398509481984
# Using the smallest frequency deviation resolution of 2^-34,
# the timestamp at which this translates to the edge cases are:

    (1 + 2**-34) * t = 2^54

Since t ~ 2^54, we have t*2^-34 ~ 2^20, so t ~ 2^54 - 2^20.
We verify:
    t               = (1<<54)-(1<<20) = 18014398508433408
    t/(1<<34)       = (1<<20)-(<1)    =           1048575.99994
    t*(1/(1<<34)+1) = (1<<54)-1       = 18014398509481983
    " + 1           =                 = 18014398509481984
    " + 1           =                 = 18014398509481985

So the transformation, under 2^-34 freq deviation, becomes:

    18014398508433408  -->  18014398509481983
    18014398508433409  -->  0
    18014398508433410  -->  1
    18014398509481983  -->
    18014398509481984  -->
    18014398509481985  -->

# Create pipes
mkfifo testfifo

# Start script (different shell)
./src/fpfind/freqcd -f 12300 -F testfifo

# Open file descriptors for write
exec 3>testfifo

# Pipe with multiple fcorr inputs
# Expected output: Final fcorr '34000' (after a broken '340''000' read)
python3 -c "print('n'.join(map(str, [1000*i for i in range(35)])) + 'n', end='')" | tr 'n' '\n' >> testfifo

# Teardown
exec 3>&-
"""