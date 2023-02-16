# Dataset Description

## 20221109
Timestamp : Freq offset (2ns peak sigma, 32ns peak sigma, time delay (ns) with 2ns resolution)\
170747 : +0Hz      (382, 441, +1_825364)\
170837 : +0.0001Hz (571, 341, +1_861040)\
171205 : +0.001Hz  (453, 349, +1_913520)\
171245 : +0.01Hz   (246, 393, +1_939790)\
171401 : +0.1Hz    (26,  387, +1_967440)\
171525 : +0.2Hz    (14,  202, +1_968784)\
171622 : +1Hz      (5,   40,  +2_000336)\
171652 : +10Hz     (5,   5,  +17_100564)\
171723 : +100Hz    (5,   5, +130_924196)\
171752 : +1000Hz   (5,   5,  -45_161958)\
171840 : -1Hz      (6,   39, -20_099074)\
171924 : -0.1Hz    (26,  350, +2_094592)\
171954 : -0.01Hz   (249, 461, +2_095062)

## 20221123
Timestamp : Freq offset\
170157 : +2000 Hz (This dataset got problem)\
165621 : +1000 Hz\
165409 : +100 Hz\
165213 : +50 Hz

## 20221209 - 20221212
6 - 13: same pair statistics, with different detuning frequencies (to verify if there is an upper limit to detuning detection)
- dataset06: (1.0000028052070675e-05, 1825518.425567852)
- dataset07: (5.000000901200785e-05, 1503675.352295511)
- dataset08: (9.999998679610655e-05, 365383.20979095786)
- dataset09: (0.00010000004867194434, 3054633.975223057)
- dataset10: (0.00010000001728416308, 3003673.092234519)
- dataset11: (0.00012000003278700433, 2756886.53014805)
- dataset12: (0.00015000001186016299, 2728534.2221366367)
- dataset13: Result doesn't make sense -- with delta_t decreasing, the peak of the cross correlation gradually becomes very weak that they are almost noise level

14 - 16: no external clock (unknown detuning)
- dataset14: (-1.3800127040219934e-05, 2408780.5806902023)
- dataset15: suspect that the dataset got some problem, as all the delta_T2 plots are empty
- dataset16: (-1.3766863843689414e-05, 2025607.5424723327)

17  - 22: pair statistics as per efficiency in table above (row starting with 093106) (to check if poor signal-noise still can detect anything, and if higher rate of coincidences will help)
- dataset17: (5.0000168822839086e-05, 1765129.0980492844)
- dataset18-22: same problem as dataset13

23 - 25: no external clock as well
- dataset23: (-1.4203862197970096e-05, 810118.604608008) -- after the freq detune is calculated, it goes in the loop to find the freq detune again, but the new frequency result is very small
- dataset24: (-1.418716948908827e-05, 723158.2112482853)
- dataset25: (-1.4211331378244374e-05, 674943.4732510289)

26 - 27: uses different sets of pair statistics (see if hyperparams needed will vary for different set of statistics)
- dataset26: (-1.4076318661349063e-05, 461557.46502057614)
- dataset27: (-1.4081502466400941e-05, 199173.13580246913)

## 20230130
28 - 30: same timestamp, thermal source\
31 - 32: different timestamp, thermal source, no external clock

## test
Using dataset 27, add time offset from 10ms to 1s