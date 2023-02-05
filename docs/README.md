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
