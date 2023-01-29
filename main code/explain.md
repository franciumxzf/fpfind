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



## Update until Dec 1st
<!-- 1. 32 bits integer (in the timestamp also) -->
1. adjust size of k (+ k*delta_t) to compensate the negative value: tried for k = 200 works quite well
2. plot the cross correlation diagram: refer to the `test` folder, `time_freq_50_new.ipynb` for the diagram with compensation, `time_freq_50.ipynb` without compensation.
3. try the python g2 code instead of fft: implemented the g2 method in `g2_two_timestamps.py`, tried but very slow...
4. By changing different Ta, Ts and N, we found that the choice of our hyperparameters can actually affect the precision of our result. Therefore, we might need a systematic way to generate appropriate hyperparameters from the dataset itself. (Need to think that how can we achieve this, as in practice we totally know nothing about our data.)\
Ongoing as I'm still not sure how to deal with this. Found that for the two 1e-5 drift datasets (10s and 100s), a set of parameter works perfectly for one but totally not working for the other. This happens for some other datasets too.\
Update: it may not help, since the data itself is also stochastic. Suspect might be mildly related to the rate of incoming timestamps as well, and the second order stability of the clock frequency (right now they're conditioned on the same local function gen)
5. Our code can solve for negative frequency shift, but can it solve for both negative frequency shift and negative time offset together?
6. Some results for now: 
    - The time and freq offset works together up to 1e-5, and looks the accuracy is quite high. The results can be found in `time_freq.py`. 
    - For 1e-4, I still believe that the algorithm can work but it doesn't for now... I found that for the 2000hz delay one alice is quite short (means after ~6*Ta it is empty), 1000hz one I plot all the cross correlation out and the peak is always not obvious; can refer to `time_freq_long.ipynb`
    - Another reason maybe is due to the memory limit that my bin number cannot go that high

## Update until Dec 8th
This week mainly works on improving the g2lib. 

Previously, the time complexity of g2 is O($n^3$). We tried to improve this by using slicing in the numpy array: `arr = arr[1:]`. By testing this method with a relatively smaller-size dataset, we obtained the same peak location as fft method, but the peak value is a bit different. (This is reasonable as the fft method involves complex number.)

Currently we are trying to employ the g2lib in `time_freq_g2.py`. An example of this can be found in `time_freq_long.ipynb`.

## Update until Dec 15th
We noticed some problems from the algorithm. Firstly, for the compensation method, different compensation actually changes the delay, and they are not really related to the compensation value. Secondly, if the frequency detune goes larger, $\Delta T_1$ and $\Delta T_2$ don't converge. In this way, we decided to give up the compensation method, and get the correct time offset value by performing one last pass of the frequency compensated dataset.

This week we mainly works on making the algorithm work for longer frequency offset.

Several new timestamp data were collected with various frequency detune to test the algorithm. More details can be found in `dataset_description.md`.

Some things we learnt from new data:
- Upper bound to algorithm with the current parameters is at least 1.5e-4, but only limited to strong peak.
- Pair statistics does strongly influence whether peak can be easily found or not. Fails to converge for 1e-4 (later found, around 5e-5) when peak signal is poor.
- Relative timestamp drift is only on the order of 2e-5, which is pfind can easily find even with poor peak.

Now, if we know the upper limit of our algorithm (say 5e-5), we can manually apply the upper limit to bob, and let it run through the algorithm again. Ideally, the frequency result will be less than the upper limit, then run the algorithm again until it gives us 0 frequency result, and we can manually calculate the total frequency shift; or if we still get a non-sense number, we can increase the correction to 2 * upper limit and so on. With the final frequency offset value, we can find the time offset.

This method generally works. The only problem is that our condition to tell whether the algorithm has found the correct frequency value is somewhat random. An improved method is purposed (see `explain.md`) and we will work on it.

For now, we aim to fully resolve the freq offset for all the datasets (provided the data is sane), fix the time delay calculation. In the meantime, we will also work on the C code.

## Update until Dec 29th
In the past two weeks, we improved the `pfind` function as described in `explain.md`. This method can do manual detune for both sides concurrently. Dataset 9 - 27 have been tested and results are acceptable. Next step is to test the algorithm on all other datasets that we have and use `generate_freqdetuneg2.py` to design the test cases around this method.

## Update until Jan 5th
The algorithm was tested with all datasets. A conservative upper limit for `time_freq` function is set to be 5e-5. The generalized result can be found in `dataset_result.md`.


1. output from pfind.py

import sys
print(..., file=sys.stderr)

2. defining the input to pfind as a command line arguments

argparse library

- import pfind (use s15 code)
- os.system: from pfind.c call pfind.py

3. use the arguments instead of the timestamp file

4. costream