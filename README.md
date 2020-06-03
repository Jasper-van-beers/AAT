# AAT

A module designed to aid the importing, filtering and processing of the Data collected through Hilmar Zech's Mobile AAT app (Zech, H.G., Rotteveel, M., van Dijk, W.W. et al. A mobile approach-avoidance task. Behav Res (2020). https://doi.org/10.3758/s13428-020-01379-3). The data of the experiment described in the aforelinked article can be found here: https://osf.io/y5b32/. This link also includes data analysis code written by Zech, so much of this module reflects what he has already done. 

# Goal
The goal of this module is to create a unified Python library which can easily handle the mobile AAT data, supplemented with documentation on how the module itself works. Moreover, some additional features will also be implemented, such as a correction for rotations of the phone. 

# Current Features
This module is still a work in progress, and not many features have been included as of yet. Currently, the module supports:
- Importing of data
- Filtering for missing data
- Filtering for unrealistic reaction times (default set to 200 ms)
- Filtering for unrealistic forces (e.g. sustained average acceleration of 5 m/s^2 over 2 seconds, which is not possible with the movements involved with the experiment)
- Offset corrections for acceleration (offset corrections, based on the assumption that the device is still at the beginning of a trial)
- Resampling of the accelerometer and gyroscope data, including a unification of their respective time arrays
- Angle corrections for acceleration, using quaternions, which occur due to (any) rotations of the phone during the trial (All rotations are made with respect to the initial orientation at the beginning of the trial, and thus represent relative angle changes). 
- Distance computations, obtained from (angled corrected) accelerometer data. NOTE: These distance computations are rather unreliable at the moment since they are dervied from the double integration of the acceleration data. Therefore, errors (from noise and bias) scale with time squared. Even though each trial only runs for ~2 seconds, the distances we are interested in a quite small (in the order of centimeters) and thus are still quite susceptible to the errors. This can be improved with addtional sensor data (e.g. magnetometer), so until this is present, the computed distances will remain unreliable. 
- The plotting of Acceleration-Time, Distance-Time, 3D trajectory, 3D acceleration, and animations of 3D acceleration and 3D trajectory
