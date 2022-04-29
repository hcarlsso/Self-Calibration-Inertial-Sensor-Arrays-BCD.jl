# Experiment information

Calibration experiments using the MIMU4444 device. The measurements were
transferred to the computer using Bluetooth, and the sampling frequency
could not be very high. The sampling frequency was set to 62.5 Hz. The
IMUs used are the 8 corner on the board. The nominal position are defined in
`IMU_positions.csv`.


The measurements are stored in the hdf format in the files
`measurements_static_and_dynamic.hdf`. Measurements were collected using two
phases; static and dynamic phases. In the static phase the array was placed in
20 different directions relative to the gravity vector, as described in the paper.

The format of the data is in matrix format with size 6K x N, where K is the
number of IMUs and N is the number of time samples. The data for IMU 1 is the
first 6 rows, where the three first rows are the accelerometer measurements
(x,y,z) [m/s^2] and the following 3 rows are the gyroscope measurements (x,y,z)
[deg/s].
