# An iteration algorithm to find the time and frequency offset
`time_freq` :

1. Set the start point at the later start of two arrays. Set the acquisition time interval $T_a$ and separation time interval $T_s$ accordingly. ($T_a = 2^{29}$, $T_s = 8*T_a$)
2. Choose the smallest discretization time $\delta t$ that is compatible with a high chance of succesfully identifying the correct peak in the cross correlation (2ns), pair with a suitable $N$ and process the two arrays $\{a_j\}$ and $\{b_k\}$ accordingly (on the first acquisitation time interval)
3. Generate the cross correlation array $\{c_k\}$, find the index $k$ of the maximum value ($\Delta T_1 = k * \delta t$) and estimate its statistical significance
4. If S is below the chosen significance limit $S_{th}$ (6), adding entries {$c_k$} pairwise (the effective $\delta t$ is doubled, update accordingly) and repeat step 3; otherwise continue
5. With the latest $\delta t$, generate discrete arrays $\{a_j\}$, $\{b_k\}$ and $\{c_k\}$ for the second acquisition time interval, and determine $\Delta T_2$
6. Determine the frequency offset: $\Delta u = \frac{\Delta T_1 - \Delta T_2}{T_s}$
7. If $\Delta u$ is smaller than 1e-10, treat it as 0 and only calculate for time offset.
8. If $\delta t$ in the last iteration is small enough, the algorithm is finished
9. Correct the event B times $\{t'_j\}$ according to $t' = (t' - \Delta T_{1 correct}) / (1 - \Delta u_{correct})$
10. Choose the same $N$, reduce the time interval $\delta t$ by $T_s/T_a/\sqrt{2}$
11. Perform the binning for the original set A and corrected set B on the two acquisition time intervals, find $\Delta T_{1 correct}$ and $\Delta T_{2 correct}$ accordingly
12. Update $\Delta T_1 = \Delta T_{1 correct} + \Delta T_1 / (1 - \Delta u_{correct})$, $\Delta T_2 = \Delta T_{2 correct} + \Delta T_2 / (1 - \Delta u_{correct})$
13. Calculate $\Delta u_{correct} = \frac{\Delta T_{1correct} - \Delta T_{2correct}}{T_s}$
14. Update $u = \frac{u}{1 - \Delta u_{correct}}$
15. $\Delta u = \frac{1}{u} - 1$
16. Continue with step 8

The above algorithm is able to calculate the correct frequency offset value (upper limit for a single pass is about $\pm 5e-5$). In order to calculate the correct time offset value, another pass with corrected bob is needed.

`pfind` :

Now, we want to improve the upper limit of the above algorithm. We proposed the following steps:
1. calculate freq offset, apply and see if the next offset calculated drops below 5e-7 (tbc? need to add another condition that smaller than the previous value); if it cannot find the peak, directly goes into step 2
2. if offset is still high => the offset calculated was not correct => apply step size / minus step size and repeat step 1 (there is a flag to decide whether we need to plus or minus)
3. For plus step size side or minus step size side, if there's one step that we already cannot find the peak, means it is the wrong direction and we won't calculate this side anymore, stick to the other side
4. set limit for both sides, if reach the limit and still cannot calculate the correct result, means that we need to change the hyperparameters
5. if offset is low => combined offset should be correct
6. do another pass, if new offset is higher than the last offset correction, then this combined offset is a false value => apply step size from initial and repeat step 1
7. if smaller, then the combined offset is correct => calculate time delay

The basic idea is we keep doing freq correction until we reach a low enough value (say 1e-9). If at any point the freq correction increases, we know the result is bogus, so we apply step size, then repeat.

`result` :

In orther to check the correctness of the result, easiest to write a test that checks for detuning between alice->bob, and bob->alice, see whether they converge and satisfy the condition:
- (alice_freq + 1) * (bob_freq + 1) = 1
- alice_time * (1 + alice_freq) + bob_time = 0