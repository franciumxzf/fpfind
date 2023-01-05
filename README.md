# Clock-sync
## data
- `1_rawevents`, `20221123`, `20221209` and `20221212` contain the detailed results
- `dataset_description.md`: descriptions of all datasets
- `dataset_result.md`: generalized results with specified parameters and remarks
- `time_freq_50.ipynb` and `time_freq_50_compensate.ipynb` show the difference of the plots when using different `find_max` function
- `time_freq_g2lib.ipynb`: plots when using `g2_two_timestamps.py` method

## freqdetuneg2 
- `parse_timestamps.py`: used to convert the raw timestamp data into numpy array
- `generate_freqdetuneg2.py`: used to artificially create the frequence drift

## main code
- `explain.md`: pseudo code to explain the algorithm
- `g2_two_timestamps.py`: taken two timestamp datasets, a slow algorithm to find the cross correlation
- `time_freq.py`: main algorithm
- `time_freq_g2.py`: main algorithm using `g2_two_timestamps.py` method (not updated as this method is very slow)

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