# Clock-sync
## freqdetuneg2 
- `parse_timestamps.py`: used to convert the raw timestamp data into numpy array
- `generate_freqdetuneg2.py`: used to artificially create the frequence drift

## random
- `explain.md`: pseudo code to explain the algorithm
- `g2_two_timestamps.py`: taken two timestamp datasets, a slow algorithm to find the cross correlation
- `time_freq.py`: main algorithm



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