# Module to handle data coming from the raw AAT files

# Use this code to debug -> Insert where we want code to pause
# import code
# code.interact(local=locals())

import pandas as pd
import numpy as np
import os
import json
import scipy.interpolate as spinter
import scipy.signal as spsig
from tqdm import tqdm

from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt
from matplotlib import animation as animation
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib as mpl

# Class to handle importing, filtering, and pre-processing of data from the AAT
class DataImporter:

    # Function to initialize the DataHandler class
    # Stores the constants used by other scripts (e.g. Dataframe column names and file paths to data)
    # Any global variables will be defined here
    def __init__(self, condition_folder_path, data_folder_path, printINFO = True):
        self.INFO = printINFO
        self.cond_path = condition_folder_path
        self.data_path = data_folder_path
        self.cwd = os.getcwd()

        self.constants = {'ACCELERATION_COLUMN' : 'acceleration',
                        'ACCELERATION_X_COLUMN' : 'acceleration_x',
                        'ACCELERATION_Y_COLUMN' : 'acceleration_y',
                        'ACCURACY_COLUMN' : 'accuracy',
                        'BLOCK_COLUMN' : "block",
                        'BUTTERED_COLUMN' : 'buttered',
                        'CONDITION_COLUMN' : "condition",
                        'CORRECT_RESPONSE_COLUMN' : 'correct_response',
                        'DEVICE_COLUMN' : 'device',
                        'DRAWN_AT_COLUMN' : 'drawn_at',
                        'DRAWN_AT_UNIX_COLUMN' : 'drawn_at_unix',
                        'EXPERIMENT_COLUMN' : 'experiment',
                        'FORCE_COLUMN' : 'force',
                        'GYRO_TIME_COLUMN' : 'gyro_times',
                        'GYRO_X_COLUMN' : 'gyro_x',
                        'GYRO_Y_COLUMN' : 'gyro_y',
                        'GYRO_Z_COLUMN' : 'gyro_z',
                        'IS_PRACTICE_COLUMN' : 'is_practice',
                        'NUMERICAL_QUESTION_TYPES' : ['likert'],
                        'CATEGORICAL_QUESTION_TYPES' : ['multiple'],
                        'PARTICIPANT_COLUMN' : 'participant',
                        'PEAK_AT_COLUMN' : 'peak_at',
                        'RESPONSE_COLUMN' : 'response',
                        'RT_COLUMN' : 'rt',
                        'RT_INV_COLUMN' : 'rt_inv',
                        'SENSOR_TYPE_COLUMN' : 'sensor_type',
                        'SESSION_COLUMN' : "session",
                        'SIGNED_UP_COLUMN' : 'signed_up',
                        'STIMULUS_COLUMN' : 'stimulus',
                        'STIMULUS_SET_COLUMN' : 'stimulus_set',
                        'TIME_COLUMN' : 'times',
                        'TRIAL_NUMBER_COLUMN' : 'block_trial',
                        'TRIAL_NUMBER_CUM_COLUMN' : 'trial',
                        'V_FINAL_COLUMN' : 'v_final',
                        'V_MAX_COLUMN' : 'v_max',
                        'ACCELEROMETER_NOISE' : 'accelerometer_noise',
                        'GYRO_NOISE' : 'gyro_noise',
                        'ANGLE_X_COLUMN' : 'angle_x',
                        'ANGLE_Y_COLUMN' : 'angle_y',
                        'ANGLE_Z_COLUMN' : 'angle_z',
                        'DISTANCE_X_COLUMN' : 'distance_x',
                        'DISTANCE_Y_COLUMN' : 'distance_y',
                        'DISTANCE_Z_COLUMN' : 'distance_z',
                        'TOTAL_DISTANCE_COLUMN' : 'total_distance'}
        standard_cols = ['PARTICIPANT_COLUMN', 'DEVICE_COLUMN', 
                        'EXPERIMENT_COLUMN', 'CONDITION_COLUMN', 
                        'SIGNED_UP_COLUMN', 'SESSION_COLUMN']
        aat_cols = ['SENSOR_TYPE_COLUMN', 'BLOCK_COLUMN', 
                    'TRIAL_NUMBER_COLUMN', 'TRIAL_NUMBER_CUM_COLUMN', 
                    'IS_PRACTICE_COLUMN', 'STIMULUS_COLUMN', 
                    'CORRECT_RESPONSE_COLUMN', 'TIME_COLUMN', 
                    'ACCELERATION_COLUMN','ACCELERATION_X_COLUMN',
                    'ACCELERATION_Y_COLUMN', 'GYRO_TIME_COLUMN', 
                    'GYRO_X_COLUMN','GYRO_Y_COLUMN',
                    'GYRO_Z_COLUMN','DRAWN_AT_UNIX_COLUMN', 'DRAWN_AT_COLUMN']
        self.cols = [self.constants[x] for x in standard_cols]
        self.aat_cols = [self.constants[x] for x in aat_cols]
        self.ExpectedParticipantData = {'participantId':'{}'.format(self.constants['PARTICIPANT_COLUMN']),
                                        'experiment':'{}'.format(self.constants['EXPERIMENT_COLUMN']),
                                        'device':'{}'.format(self.constants['DEVICE_COLUMN']),
                                        'signed_up':'{}'.format(self.constants['SIGNED_UP_COLUMN']),
                                        'sensor_type':'{}'.format(self.constants['SENSOR_TYPE_COLUMN']),
                                        'condition':'{}'.format(self.constants['CONDITION_COLUMN']),
                                        'completion':'completion'}
        self.ExpectedBlockData = {'CORRECT_RESPONSE_COLUMN':'correctResponse',
                                  'STIMULUS_COLUMN':'imageName',
                                  'DRAWN_AT_UNIX_COLUMN':'drawnAtUnix',
                                  'DRAWN_AT_COLUMN':'drawnAt'}
        self.ExpectedNumericData = {'acceleration_z':'acceleration', 
                                    'acceleration_x':'acceleration_x',
                                    'acceleration_y':'acceleration_y',
                                    'gyro_x':'gyro_x',
                                    'gyro_y':'gyro_y',
                                    'gyro_z':'gyro_z'}
        self.ExpectedNumericData2Constants = {}
        inv_ExpectedNumericData = {v: k for k, v in self.ExpectedNumericData.items()}
        for key in self.constants.keys():
            if self.constants[key] in self.ExpectedNumericData.values():
                self.ExpectedNumericData2Constants.update({'{}'.format(inv_ExpectedNumericData[self.constants[key]]):key})

        # Which data columns to check for NaN values
        self.DataCols2Check = ['ACCELERATION_COLUMN', 'ACCELERATION_X_COLUMN', 'ACCELERATION_Y_COLUMN',
                               'TIME_COLUMN']

        # Inverted Data columns to resample
        self.Data2Resample = {'acceleration_z':'ACCELERATION_COLUMN',
                        'acceleration_x':'ACCELERATION_X_COLUMN',
                        'acceleration_y':'ACCELERATION_Y_COLUMN',
                        'gyro_x':'GYRO_X_COLUMN',
                        'gyro_y':'GYRO_Y_COLUMN',
                        'gyro_z':'GYRO_Z_COLUMN'}


    # Basic utility function to check if specified files are found in the specified directory
    # Input:    files = list of strings containing the filenames to check
    #           path = path to the folder where files are expected
    # Output:   output = Boolean indicating if files are present or not. 
    def CheckFiles(self, files, path):
        os.chdir(path)
        output = all(os.path.isfile(os.path.join(path, file)) for file in files)
        os.chdir(self.cwd)
        return output

    
    # Utility function to save dataframes for future use
    # Input:    filename = The filename of the to be saved dataframe 
    #           dataframe = Dataframe to be saved
    #           path = Save path
    # Output:   None
    def SaveDF(self, filename, dataframe, path):
        os.chdir(path)
        dataframe.to_pickle('./{}'.format(filename))
        os.chdir(self.cwd)
        return None



    # Utility function to load saved dataframes
    # Input:    filename = The name of the file to be loaded
    #           path = Path to the file
    # Output:   df = DataFrame containing data from filename
    def LoadDF(self, filename, path):
        os.chdir(path)
        df = pd.read_pickle(os.path.join(path, filename))
        os.chdir(self.cwd)
        return df



    # TODO: Utility function to rename columns of the columns of the dataframe and modify
    # the appropriate global class constants
    def RenameDFCols(self):
        # Add something to rename acceleration column to acceleration_z
        pass


    
    # Function to compute distances, based on the accelerometer data. 
    # NOTE: With current information, reliable estimates of distance cannot be obtained. 
    # Obtaining distance is not so trivial -> Due to double integration, noise and bias 
    # cause large errors in estimates (scale with time squared).
    # https://www.youtube.com/watch?v=C7JQ7Rpwn2k @ 25:22 shows that, in 1 second, we can 
    # have drift of up to 20 cm
    # -> Errors in orientation estimatation are an order of magnitude worse -> 1 degree off 
    #    could lead to several meters of error in distance.
    #    https://www.cl.cam.ac.uk/techreports/UCAM-CL-TR-696.pdf
    # -> Corrections to gyroscopic data can also be used with the gravity vector, if known
    #       -> Current AAT does not store gravity vector
    # -> Corrections can be made with additional sensor information (e.g. magnetometer)
    #       -> Current AAT does not store additional sensor data 
    # With an additional sensor, a kalman filter may be used to estimate distance. 
    # Input:    DataFrame = (Angle Corrected) DataFrame containing AAT data
    #           ComputeTotalDistance = Boolean to toggle the computation (and 
    #                                  storage) of the total distance; i.e. magnitude of
    #                                  displacement per trial. Default = False, meaning 
    #                                  that total distances will not be computed. 
    # Output:   DF_out = The same as the input DataFrame, but with new columns containing
    #                    the computed displacements (distances). 
    def ComputeDistance(self, DataFrame, ComputeTotalDistance = False):

        # Function to integrate the acceleration data to obtain velocities and 
        # distances
        def integrate_acceleration(a_vec, t, scale = 1000.):
            # Scale time (default time is in milliseconds -> convert to seconds)
            t = t/scale
            # Preallocate variables
            v_vec = np.copy(a_vec) * 0
            d_vec = np.copy(a_vec) * 0
            # Using physical equations (assume that acceleration for one time-step, dt, is constant)
            # integrate the acceleration to get velocity, then integrate again to get displacement
            for i in range(len(t)-1):
                # Infer time step
                dt = t[i + 1] - t[i]
                # Compute velocity
                v_vec[:, i + 1] = v_vec[:, i] + a_vec[:, i] * dt
                # Comput displacement 
                d_vec[:, i + 1] = d_vec[:, i] + v_vec[:, i] * dt + 0.5 * a_vec[:, i] * dt ** 2 
            return d_vec, v_vec

        DF = DataFrame.copy(deep = True)
        
        # If the user decided to display information to the terminal window
        if self.INFO:
            print('[INFO] Computing distances...')
            iterrows = tqdm(DF.iterrows(), total=DF.shape[0])
        else:
            iterrows = DF.iterrows()

        DF_out_list = []

        for idx, row in iterrows:
            # Calculation of Distance and Velocity
            # Create acceleration vector
            _acc_vec = np.vstack((row[self.constants['ACCELERATION_X_COLUMN']],
                    row[self.constants['ACCELERATION_Y_COLUMN']],
                    row[self.constants['ACCELERATION_COLUMN']]))
            _time_vec = row[self.constants['TIME_COLUMN']]
            _distance, _velocity = integrate_acceleration(_acc_vec, _time_vec)
            # If user opted to compute total distance
            if ComputeTotalDistance:
                total_distance = np.sqrt(np.sum(np.square(_distance), axis = 0))

            # Saving data to new dataframe
            DataRow = row.copy(deep = True)

            # Store data
            DataRow[self.constants['DISTANCE_X_COLUMN']] = _distance[0, :]
            DataRow[self.constants['DISTANCE_Y_COLUMN']] = _distance[1, :]
            DataRow[self.constants['DISTANCE_Z_COLUMN']] = _distance[2, :]
            if ComputeTotalDistance:
                DataRow[self.constants['TOTAL_DISTANCE_COLUMN']] = total_distance
            
            DF_out_list.append(DataRow)

        # Compile new dataframe with distance data
        DF_out = pd.concat(DF_out_list, axis = 1).transpose()

        return DF_out



    # Function to apply the rotation correction for acceleration to the entire
    # DataFrame
    # Input:    DataFrame = (Resampled) DataFrame containing the AAT data
    #           StoreTheta = Boolean to determine if the angles should be stored
    #                        in the output dataframe (True -> angles will be 
    #                        stored)
    # Output:   CorrectedData = A DataFrame resembling the input DataFrame, but
    #                        with rotation correct accelerations and (if set)
    #                        the corresponding rotation angles
    def Correct4Rotations(self, DataFrame, StoreTheta = True):
        DF = DataFrame.copy(deep = True)

        # If the user decided to display information to the terminal window
        if self.INFO:
            print("[INFO] Correction for rotations...")
            iterrows = tqdm(DF.iterrows(), total=DF.shape[0])
        else:
            iterrows = DF.iterrows()
        
        # Preallocate list to store corrected data, which will later be compiled
        # into the output dataframe
        CorrectedDataList = []

        # Define intial angle (assume 0 since device is stationary)
        # Even if the device is tilted, setting the intial angle to zero will
        # give the change in angle relative to this position
        InitialTheta = np.zeros((3, 1))
        # Set angle threshold, above which a rotation is considered to be occuring
        AngleThreshold = 1*np.pi/180.

        for idx, row in iterrows:
            # If there is gyroscopic data available, compute rotations
            try: 
                if len(row[self.constants['GYRO_X_COLUMN']]):
                    # Create angular rate vector
                    _gyro_vec = np.vstack((row[self.constants['GYRO_X_COLUMN']],
                                        row[self.constants['GYRO_Y_COLUMN']],
                                        row[self.constants['GYRO_Z_COLUMN']]))
                    # Create acceleration vector
                    _acc_vec = np.vstack((row[self.constants['ACCELERATION_X_COLUMN']],
                                        row[self.constants['ACCELERATION_Y_COLUMN']],
                                        row[self.constants['ACCELERATION_COLUMN']]))
                    # Integrate angular rates to get angles
                    _th_vec = self._IntegrateGyro(row[self.constants['TIME_COLUMN']], _gyro_vec, InitialTheta, scale = 1000)
                    # Apply rotation correction to the acceleration data 
                    _CAcc_vec = self._ApplyRotationCorrection(_acc_vec, _th_vec, AngleThreshold=AngleThreshold)
                    DataRow = row.copy(deep = True)

                    # Replace old acceleration values with corrected acceleration values 
                    DataRow[self.constants['ACCELERATION_X_COLUMN']] = _CAcc_vec[0, :]
                    DataRow[self.constants['ACCELERATION_Y_COLUMN']] = _CAcc_vec[1, :]
                    DataRow[self.constants['ACCELERATION_COLUMN']] = _CAcc_vec[2, :]

                    # If the user opted to save the angles, add these to new 'angle' columns
                    # in the dataframe
                    if StoreTheta:
                        DataRow[self.constants['ANGLE_X_COLUMN']] = _th_vec[0, :]
                        DataRow[self.constants['ANGLE_Y_COLUMN']] = _th_vec[1, :]
                        DataRow[self.constants['ANGLE_Z_COLUMN']] = _th_vec[2, :]
                # For cases where there is no gyroscopic data, no angular corrections can be made
                # (Since the gravity vector is also not known)    
                else:
                    if StoreTheta:
                        DataRow[self.constants['ANGLE_X_COLUMN']] = np.nan
                        DataRow[self.constants['ANGLE_Y_COLUMN']] = np.nan
                        DataRow[self.constants['ANGLE_Z_COLUMN']] = np.nan
                    DataRow = row.copy(deep = True)
                    
            except TypeError:
                    if StoreTheta:
                        DataRow[self.constants['ANGLE_X_COLUMN']] = np.nan
                        DataRow[self.constants['ANGLE_Y_COLUMN']] = np.nan
                        DataRow[self.constants['ANGLE_Z_COLUMN']] = np.nan
                    DataRow = row.copy(deep = True)

            CorrectedDataList.append(DataRow)

        # Compile new dataframe with corrected data
        CorrectedData = pd.concat(CorrectedDataList, axis = 1).transpose()
        
        return CorrectedData



    # Function to correct the accelerations for rotations during the response of a single 
    # participant, based on information from the gyroscopic data, using quaternions. 
    # Input:    a_vec = acceleration vector (3 x N) where N is the number of samples
    #           th_vec = angular vector (3 x N) 
    #           AngleThreshold = Minimum angle above which a rotation is considered
    #                            to have occured. This is necessary since the order
    #                            of rotations is important. Default is 1 degree, 
    #                            input should be in radians
    # Output:  a_corrected = Rotation corrected acceleration vector (3 x N) 
    def _ApplyRotationCorrection(self, a_vec, th_vec, AngleThreshold = 1*np.pi/180):
        # As order of rotations is important, infer the order of rotations
        # based on which axes first exceed the 'AngleThreshold' 
        # The default angle threshold is set to 1 degree, as below this the small
        # angle approximation typically holds (i.e. rotations are negligible)
        # If angle changes are not significant, then assume an order of x->y->z
        try:
            IndexExceedingThres =  [(np.where(abs(th_vec[0]) > AngleThreshold)[0][0]),
                                    (np.where(abs(th_vec[1]) > AngleThreshold)[0][0]),
                                    (np.where(abs(th_vec[2]) > AngleThreshold)[0][0])]
            # We reverse the order of the argsort since argsort will give the highest
            # value first, whereas we want the lowest values (i.e. first instance)
            RotOrder = np.argsort(IndexExceedingThres)[::-1]
        except IndexError:
            RotOrder = np.array([2, 1, 0])

        # Convert angualr vector into quaternion vector
        quat_vec_original = self._GetQuat(th_vec)
        # We need to re-arrange the quaternion vector to correspond to the rotation
        # vector (note, quaternion variable in index [0, :] remains fixed)
        quat_vec = np.copy(quat_vec_original)
        for i in range(len(RotOrder)):
            quat_vec[i + 1] = quat_vec_original[RotOrder[i] + 1]
        
        # Since we are going from the body, to the inertial frame, we need to take the inverse of the
        # quaternion, q (which is equivalent to its conjugate). We are going from the body frame to the
        # interial frame since the accelerations are measured w.r.t to the device (i.e. body frame)
        # hence, to account for any rotations (w.r.t the initial position, which we interpret as the 
        # inertial frame) we need to project our accelerations from the body frame to the inertial frame.
        q0 = quat_vec[0]
        q1 = quat_vec[1]
        q2 = quat_vec[2]
        q3 = quat_vec[3]

        # Define rotation matrices for each axis 
        R_1 = np.array([(q0*q0 + q1*q1 - q2*q2 -q3*q3), (2*(q1*q2 - q0*q3)), (2*(q0*q2 + q1*q3))])
        R_2 = np.array([(2*(q1*q2 + q0*q3)), (q0*q0 - q1*q1 + q2*q2 - q3*q3), (2*(q2*q3 - q0*q1))])
        R_3 = np.array([(2*(q1*q3 - q0*q2)), (2*(q0*q1 + q2*q3)), (q0*q0 - q1*q1 - q2*q2 + q3*q3)])

        # Manipulate the indices of the rotation matrices above to get a vector of form
        # N x [3 x 3] such that each element corresponds to the rotation matrix for that
        # specific sample and can therefore be multiplied directly with the acceleration array
        R_1 = R_1.T
        R_2 = R_2.T
        R_3 = R_3.T
        R_stack = np.zeros((3*len(R_1), 3))
        R_stack[0:(3*len(R_1)):3] = R_1
        R_stack[1:(3*len(R_1)):3] = R_2
        R_stack[2:(3*len(R_1)):3] = R_3
        R = R_stack.reshape((len(R_1), 3, 3))
        
        # Transpose the acceleration vector such that the matrix multiplication agrees
        a_vecT = a_vec.T
        # Correct for the accelerations
        a_corrected = np.matmul(R, a_vecT.reshape((len(R_1), 3, 1))).T
        a_corrected = a_corrected.reshape(a_vec.shape)

        return a_corrected



    # Function to get quaternion representation of the angle vector
    # Input:    theta_vec = Angular vector (3 x N) where N is the number of samples
    # Output:   quat_vec = Quaternion equivalent (4 x N) of theta_vec 
    def _GetQuat(self, theta_vec):
        # Preallocate quaternion vector
        quat_vec = np.zeros((4, len(theta_vec[0])))

        # For each time step
        for t in range(len(theta_vec[0])):
            # Compute the angular magnitude
            mag = np.sqrt( (theta_vec[0, t]**2 + theta_vec[1, t]**2 + theta_vec[2, t]**2) )

            # Avoid division by 0 for 0 magnitude, which would occur if all angles are 0. 
            if theta_vec[:, t].all() == 0 and mag == 0:
                mag = 1

            # Normalize the angle vector
            Nth_vec = theta_vec[:, t]/mag

            thetaOver2 = mag/2
            sinTO2 = np.sin(thetaOver2)
            cosTO2 = np.cos(thetaOver2)

            # Convert to quaternions 
            quat_vec[0][t] = cosTO2
            quat_vec[1][t] = sinTO2 * Nth_vec[0]
            quat_vec[2][t] = sinTO2 * Nth_vec[1]
            quat_vec[3][t] = sinTO2 * Nth_vec[2]

        return quat_vec



    # Function to integrate the gyroscopic angular rates to obtain the rotation
    # angles
    # Ideally, we would use some from of state estimation (e.g. Kalman filter) to
    # obtain a better representation of the true angular rates (since integrating
    # gyroscopic data incurs significant errors, if drift and biases are not
    # accounted for). However, there is insufficient information to adequately do
    # this at this point. The time scales we are looking at are small 
    # (~ 2 seconds), so errors will not have propagated so much. That said, 
    # depending on the sensor, errors can be in the order of centimeters after even
    # just a second, which is problematic for the distances involved with the AAT
    # (~20 cm)
    # Input:    time = Time array
    #           g_vec = Angular rates array
    #           theta_ic = Initial rotation angles
    #           scale = Scale of time array, default = 1000 (i.e. time array
    #                   is given in milliseconds)
    # Output:   th_vec = Angles, of the same size as time array
    def _IntegrateGyro(self, time, g_vec, theta_ic, scale = 1000):

        # Preallocate the angular array, and add initial conditions
        th_vec = np.zeros((3, len(time))) + theta_ic

        # Integrate 
        for t in range(len(time) - 1):
            # Infer time step
            dt = (time[t + 1] - time[t])/scale
            # For each axis, integrate the corresponding angular rate using
            # Forward Euler Integration
            for i in range(len(th_vec)):
                th_vec[i, t + 1] = th_vec[i, t] + g_vec[i, t] * dt
        
        return th_vec



    # Function to resample acceleration and gyroscopic data from an inconsistent 4-5 ms
    # to 1 ms. Moreover, this function also resamples and unifies the time arrays between
    # the gyroscope and accelerometer.
    # Input:    DataFrame = (Filtered) dataframe containing the AAT data
    # Output:   ResampledDF = The same as DataFrame, but with resampled data and a unified
    #                         time array (gyro time array column now removed)
    def ResampleData(self, DataFrame):
        DF = DataFrame.copy(deep = True)
        
        # If the user decided to display information to the terminal window
        if self.INFO:
            print("[INFO] Running Resampling...")
            iterrows = tqdm(DF.iterrows(), total=DF.shape[0])
        else:
            iterrows = DF.iterrows()

        ResampledList = []

        # Resample the rows. The columns specified by DataCols2Resample in __init__()
        # will be resampled
        for i, row in iterrows:
            _ResampledRow = self._ResampleRow(row)
            ResampledList.append(_ResampledRow)

        # Recompile the dataframe with the resampled data
        ResampledDF = pd.concat(ResampledList, axis = 1).transpose()
        return ResampledDF



    # Function which is used by ResampleData. This function actually handles the 
    # computations for the resampling, while ResampleData applies this to the entire
    # data frame
    # Input:    DataRow = Row from a DataFrame containing the (filtered) AAT data
    # Output:   ResampledRow = The same as DataRow, but with a unified Time column 
    #                          (i.e. dropped gyro_time_column), and with resampled
    #                          data defined in Data2Resample.
    def _ResampleRow(self, DataRow):
        ResampledRow = DataRow.copy(deep = True)
        # Get a unified time array between the accelerometer and gyroscopic time
        ResampledTime = self._AlignTimeArrays(DataRow[self.constants['TIME_COLUMN']], DataRow[self.constants['GYRO_TIME_COLUMN']], dt = 1)
        ResampledRow[self.constants['TIME_COLUMN']] = ResampledTime
        for key, value in self.Data2Resample.items():
            # For accelerations
            if key.startswith('acceleration'):
                # Attempt to resample the acceleration data
                try:
                    ResampledRow[self.constants[value]] = self._Interpolate(DataRow[self.constants[value]], DataRow[self.constants['TIME_COLUMN']], ResampledTime)
                except TypeError:
                    pass
            # For gyroscopic data
            elif key.startswith('gyro'):
                # Attempt to resample the gyroscopic data
                try:
                    ResampledRow[self.constants[value]] = self._Interpolate(DataRow[self.constants[value]], DataRow[self.constants['GYRO_TIME_COLUMN']], ResampledTime)
                except TypeError:
                    pass
            else:
                pass
        # Remove Gyro time column as it is now redundant 
        ResampledRow.drop(self.constants['GYRO_TIME_COLUMN'])
        return ResampledRow
    


    # Function to Interpolate a data array using B-Spline interpolation. 
    # Input:    y = Data array to Interpolate
    #           x1 = Time array corresponding to y
    #           x2 = Resampled time array
    # Output:   y2 = Resmapled and _Interpolated data array
    def _Interpolate(self, y, x1, x2):
        # Scan input data for nans, remove them from the array. The best we can do is Interpolate these points with neighbors
        y_nonan = y[np.invert(np.isnan(y))]
        # Remove Y_nan indices from x-array to ensure correspondence between the two arrays
        x1 = x1[np.invert(np.isnan(y))]
        # Remove (remaining) nans from x array
        x1_nonan = x1[np.invert(np.isnan(x1))]
        # Remove x_nan indices from y-array
        y_nonan = y_nonan[np.invert(np.isnan(x1))]

        # Create b-spline interpolation
        Bspline = spinter.splrep(x1_nonan, y_nonan)
        # Interpolate new y values based on sampling points x2
        y2 = spinter.splev(x2, Bspline)

        return y2



    # Function to align the time arrays of the gyroscopic and accelerometer 
    # sensors. dt is in milliseconds 
    # Input:    acc_time = Time array corresponding to the accelerometer data
    #           gyro_time = Time array corresponding to the accelerometer data
    #           dt = Desired output time step of unified time array, in milliseconds
    # Output:   time_array = Unified time array, with time step of dt
    def _AlignTimeArrays(self, acc_time, gyro_time, dt = 1):
        # All experiments should have accelerometer data, but in case they do not
        try:
            if len(acc_time) & len(gyro_time):
                # Some devices (using Simple Accelerometer) do not have gyroscopic data
                # Find the starting time, based on the lowest common time between the 
                # two arrays
                t_start = np.nanmax((acc_time[0], gyro_time[0]))
                # Find the ending time, based on the highest common time between the 
                # two arrays
                t_end = np.nanmin((acc_time[-1], gyro_time[-1]))
                # Create a new time array spanning the time interval, in steps of dt
                time_array = np.arange(t_start, t_end, dt)
            elif len(acc_time):
                t_start = acc_time[0]
                t_end = acc_time[-1]
                time_array = np.arange(t_start, t_end, dt)
            else:
                time_array = np.nan
        except TypeError:
            if len(acc_time):
                t_start = acc_time[0]
                t_end = acc_time[-1]
                time_array = np.arange(t_start, t_end, dt)
            else:
                time_array = np.nan

        return time_array



    # Function to filter missing data (i.e. NaN fields)
    # Input:    DataFrame = DataFrame containing the (raw) AAT data
    # Output:   DF = Filtered DataFrame with NaN data removed
    def FilterNaNs(self, DataFrame):
        DF = DataFrame.copy(deep = True)

        N = DF.shape[0]

        # Find NaN values in the relevant columns (DataCols2Check) of the dataframe
        # Most important columns are accelerations and time (corresponding to acceleration)
        # Gyroscopic data is not included by default since not all devices contain such information
        IsNaNDF = DF[[self.constants[column] for column in self.DataCols2Check]].isnull()
        try:
            # Find unique indices of these NaN values
            Indices_to_remove_NaN = np.unique(np.where(IsNaNDF)[0])
        # In case there are no NaN values, return empty list of indices to remove
        except IndexError:
            Indices_to_remove_NaN = []

        # If the user decided to display information to the terminal window
        if self.INFO:
            print('[INFO] Filtering results:')
            print('[INFO] \t Percentage of data which is missing: {}'.format(len(Indices_to_remove_NaN)/N*100))

        # Remove rows with NaN values 
        DF = DF.drop(index = Indices_to_remove_NaN)
        DF = DF.reset_index(drop = True)

        return DF



    # Function to compute the reaction times of the participants, based on the acceleration signals 
    # exceeding some defined threshold. 
    # NOTE: Compute RT can be done before orientation corrections and acceleration calibration
    # because we are looking for a reaction (i.e. change in acceleration).
    # The orientation changes are given with respect to the initial orientation (since, with current
    # information, precise orientation w.r.t gravity is unknown). Hence, it is assumed that the
    # change in angles is ~0 before a reaction. 
    # Input:    DataFrame = Dataframe holding the (NaN removed) AAT data
    #           RTResLB = Lowerbound of acceleration, above which motion is considered a 
    #                     'reaction'
    #           RTHeightThresRatio = Ratio of maximum acceleration.
    #                                NOTE: In this function, we take 
    #                                max(RTResLB, RTHeightThresRatio) as the minimum peak
    #                                height
    #           RTPeakDistance = Minimum allowed distance between peaks (in milliseconds)
    # Output:   DF = Dataframe with reaction times (returns a copy of input DataFrame but
    #                with a new column for the reaction times)    
    def ComputeRT(self, DataFrame, RTResLB = 1, RTHeightThresRatio = 0.3, RTPeakDistance = 10):
        DF = DataFrame.copy(deep = True)
        # If the user decided to display information to the terminal window
        if self.INFO:
            print("[INFO] Computing Reaction Times...")
            iterrows = tqdm(DF.iterrows(), total=DF.shape[0])
        else:
            iterrows = DF.iterrows()

        # Define constants and preallocate empty Dataframe series to host RTs
        N = DF.shape[0]
        RTs = pd.Series(index = np.arange(0, N), name = self.constants['RT_COLUMN'])

        for i, row in iterrows:
            # Some signals have a lot of jitter at the beginning with relatively high accelerations. To reduce the number of participants
            # falsely identified as having too fast reaction times, we will consider any peak larger than RTHeighThresRatio of the highest 
            # peak as a significant peak. However, there are also participants with no reactions, in which case we take the absolute 
            # lowerbound, defined by RTResLB. 
            RTHeightThres = max(max(abs(row[self.constants['ACCELERATION_COLUMN']]))*RTHeightThresRatio, RTResLB)
            # The try-except catch below is to account for no reactions (which may not exceed the reaction threshold)
            try:
                # Find the first index where the acceleration exceeds this threshold
                RT_idx = np.where(abs(row[self.constants['ACCELERATION_COLUMN']]) >= RTHeightThres)[0][0]
                # Find the corresponding index in the time array
                RTs[i] = row[self.constants['TIME_COLUMN']][RT_idx]
            except IndexError:
                RTs[i] = 'No Reaction'
        DF[self.constants['RT_COLUMN']] = RTs
        return DF



    # Function to check if the reaction time is realistic (default: RT is realistic if RT > 200 ms)
    # Input:    DataRow = Row of the Dataframe contaning the AAT data
    #           RTThreshold = The cutoff reaction time, in milliseconds
    # Output:   RTisRealistic = True/False indicating if the reaction time is realistic 
    #                           (Realistic RT -> True)
    #           NoReactionFlag = True/False indiciting if the there is no reaction 
    #                            (No reaction -> True)
    def _HasRealisticRT(self, DataRow, RTThreshold):
        NoReactionflag = 0
        try:
            # Check for 'No reaction' which is a string, all other RTs should be numeric
            RT = DataRow['rt'] + 0
            if RT > RTThreshold:
                RTIsRealistic = True
            else:
                RTIsRealistic = False
        except TypeError:
            NoReactionflag = 1
            RTIsRealistic = False

        return RTIsRealistic, NoReactionflag



    # Function which estimates the offset and noise characteristics of the signal based on the
    # stationary period of time before a reaction occurs. The underlying assumption is that the
    # the acceleration before a motion should be zero and that no rotations are occuring
    # (since the device is stationary).
    # Inputs:   Row = DataRow of the dataframe containing the acceleration data
    # Outputs:  DataRow = Updated DataRow with the calibrated (offset adjusted) accelerations
    #                     and Gyroscopic data
    #           stds = Standard deviations of acceleration and noise (i.e. estimates of the 
    #                  sensor noise characteristics)
    def _CalibrateData(self, Row):
        DataRow = Row.copy(deep = True)
        
        a_std = np.nan
        g_std = np.nan

        # Extract mean and standard deviation from an input signal (indicated by column) 
        # starting from index = 0 until 'end_idx'
        def Calibrate(Data, column, end_idx):
            x = Data[self.constants[column]][0:end_idx]
            std = np.nanstd(x)
            offset = np.nanmean(x)
            return offset, std

        # Extract reaction time (to get period in which there is 'no reaction')
        rt = DataRow[self.constants['RT_COLUMN']]
        
        # Offset correct the acceleration, and extract noise characteristics
        time_a = DataRow[self.constants['TIME_COLUMN']]
        try:
            # Find the indices which correspond to the stationary period before
            # the reaction
            idx_rt_a = np.where(time_a < rt)[0][-1]
        except TypeError:
            # In some cases, there is no reaction, so default to the entire
            # signal length
            idx_rt_a = len(time_a) - 1

        # Given that the RT is not the first index of the acceleration array
        if not idx_rt_a == 0:
            a_cols = ['ACCELERATION_X_COLUMN', 'ACCELERATION_Y_COLUMN', 'ACCELERATION_COLUMN']
            a_std = {}
            for col in a_cols:
                # Compute offset and noise estimate
                offset, std = Calibrate(DataRow, col, idx_rt_a)
                # Correct the acceleration
                DataRow[self.constants[col]] -= offset
                # Extract the 'axis' label 
                key = col.split('_')[1]
                # Special handle for the 'Z' acceleration (since it is called acceleration)
                if len(key) > 1:
                    key = 'Z'
                # Store noise values 
                a_std.update({key:std})

        # If gyroscopic data exists
        try:
            # Condition below will raise a TypeError if Gyro time = np.nan or will
            # not execute if gyro type = []
            if len(DataRow[self.constants['GYRO_TIME_COLUMN']]) > 0:
                # Extract the gyroscopic time (which may be different from the accelerometer time)
                time_g = DataRow[self.constants['GYRO_TIME_COLUMN']]
                try:
                    # Find the indices which correspond to the stationary period before
                    # the reaction
                    idx_rt_g = np.where(time_g < rt)[0][-1]
                except TypeError:
                    # In some cases, there is no reaction, so default to the entire
                    # signal length
                    idx_rt_g = len(time_g) - 1

                # Given that the RT is not the first index of the acceleration array
                if not idx_rt_g == 0:
                    g_cols = ['GYRO_X_COLUMN', 'GYRO_Y_COLUMN', 'GYRO_Z_COLUMN']
                    g_std = {}
                    for col in g_cols:
                        # Compute offset and noise estimate
                        offset, std = Calibrate(DataRow, col, idx_rt_g)
                        # Correct gyroscopic data with offset
                        DataRow[self.constants[col]] -= offset
                        # Extract the 'axis' label 
                        key = col.split('_')[1]
                        g_std.update({key:std})
        except TypeError:
            pass

        stds = [a_std, g_std]
        
        return DataRow, stds



    # Function to determine if the acceleration data is realistic. Here, an approximation is made on the 
    # average acceleration needed to fully extend a human arm in the timeframe (~2 seconds). This is itself 
    # an (exaggerated) approximation of the motion of the experiment. For starters, this average acceleration
    # relies on the exact length of one's arm. Moreover, it takes the full arm distance when it is likely that
    # the true distance is less than this. However, the largest distance was taken to allow for more flexibility
    # in the conditions. In essense, this function compares the average measured accelerations, to the expected
    # average. Should the measured accelerations exceed this expected average, then the accelerations are deemed
    # unrealistic. The initial goal of this function was to remove participants who had sustained accelerations 
    # between 1-10 m/s^2 for the entire 2 seconds, which is unrealistic. 
    # NOTE: This function is not really recommended since it removes a lot of the data. This is not due
    # to particularily stringent conditions, but rather due to the inherent unreliability of the mobile 
    # sensors themselves. Another possibility is to relax these conditions, e.g. increase tolerance
    # Inputs:   DataRow = Dataframe row containing acceleration information for a given trial
    #           MaxArmDist = Maximum (Exaggerated) Arm Distance, in meters, that a participant is expected 
    #                        to cover in the AAT motion.
    #           TimeScale = The scale of time in DataRow, default is milliseconds (= 1000) 
    #           Tolerance = Additional tolerance for realistic acceleration condition. Increasing this
    #                       tolerance relaxes the cutoff for realistic accelerations
    # Output:   AccIsRealistic = Boolean, True if acceleration is considered realistic, False otherwise
    def _HasRealisticAccelerations(self, DataRow, MaxArmDist, TimeScale = 1000, Tolerance = 0.50):
        MaxDist = MaxArmDist[DataRow['gender']]
        # Add catch for any missing data (i.e. np.nans)
        try:
            # Get time window
            delta_t = (DataRow[self.constants['TIME_COLUMN']][-1] - DataRow[self.constants['TIME_COLUMN']][0])/TimeScale
            # Infer average acceleration to cover distance
            a_limit = 2*MaxDist/(delta_t**2)
            
            # Extract accelerations in all directions
            ax = DataRow[self.constants['ACCELERATION_X_COLUMN']]
            ay = DataRow[self.constants['ACCELERATION_Y_COLUMN']]
            az = DataRow[self.constants['ACCELERATION_COLUMN']]

            # Extract total average acceleration
            a_tot = np.sqrt((np.nanmean(ax)**2 + np.nanmean(ay)**2 + np.nanmean(az)**2))
            
            # If acceleration is lower than, or equal to, the limit acceleration then assume that
            # the recorded accelerations are realistic
            if a_tot <= (1+Tolerance)*a_limit:
                AccIsRealistic = True
            else:
                AccIsRealistic = False
        # If data contains nans, then assume unrealistic acceleration
        except TypeError:
            AccIsRealistic = False
        
        return AccIsRealistic



    # A general function which aggregates some of the filtering functions into one function.
    # This function allows for the filtering of:
    #   - Unrealistic Reaction times (RT)
    #   - Unrealistic Accelerations
    # Moreover, this function allows for the "calibration" of the acceleration; this involves 
    # making an estimate of the offsets in the acceleration. It is assumed that the participants
    # are holding their phone still before a reacting. Due to this, the accelerations should be 
    # near zero. Hence, any accelerations measured here are expected to be due to the offsets of 
    # the accelerometers (magnitudes are the of the expected level ~0.3 m/s^2). Hence, 'calibration'
    # involves removing these offsets from the measured data. 
    # This function allows for the toggling of the different filtering functions.
    # Moreover, the reaction time threshold can also be set (in milliseconds) and the arm lengths (MaxArmDist)
    # can also be defined; split between female and male participants (since the paper below has suggested
    # that there is a significant difference between the genders)
    # Average human reaction time to visual stimulus is 0.25 seconds -> set default threshold to 200
    # Human arm lengths https://www.researchgate.net/figure/ARM-LENGTH-IN-MM-BY-SEX-AND-AGE_tbl3_10567860
    # Inputs:   Data = Raw AAT dataframe
    #           MinDataRatio = Minimum ratio of valid trials. If a participant has less valid trials
    #                               all of their data will be removed. 
    #           RemoveRT = if True, filter unrealistically fast reaction times
    #           KeepNoRT = if True, keep 'No reactions' (i.e. max acceleration < 1m/s^-2)
    #           RTThreshold = 200, Reaction time in ms definining the cutoff for unrealistic accelerations
    #           CalibrateAcc = if True, the acceleration offsets will be taken into acocunt
    #           RemoveAcc = if True, Remove unrealistic accelerations based on the average acceleration
    #                       see function '_HasRealisticAccelerations()' for detailed summary on how this is done
    #           MaxArmDist = dictionary of the average arm lengths of adult males and females (used to estimate
    #                        average reasonable acceleration)
    # Outputs:  DF = Filtered dataframe
    #           Removed_data = Dataframe containing the removed data
    def FilterData(self, Data, MinDataRatio = 0.8, RemoveRT = True, KeepNoRT = False, RTThreshold = 200, CalibrateAcc = True, RemoveAcc = True, MaxArmDist = {'female':0.735, 'male':0.810}):
        # Make a copy of the data frame (such that we do not make changes to the input data frame directly)
        DF = Data.copy(deep = True)

        # Check if reaction times have been previously computed. Otherwise, compute the reaction times 
        try:
            if Data['rt'][0] > 0:
                pass
        except KeyError:
            print('[WARNING] The inputted DataFrame for <FilterData> does not have a reaction time column.\n\tI will try to compute reaction times here.\n\tNOTE: Default values are being used.')
            DF = self.ComputeRT(DF)

        # If the user as opted to display information
        if self.INFO:
            print("[INFO] Running Filtering...")
            iterrows = tqdm(DF.iterrows(), total=DF.shape[0])
        else:
            iterrows = DF.iterrows()

        # Initialize lists and constants
        N = DF.shape[0]
        # Lists to store the indices, of DF, which contain data which should be removed. 
        Indices_to_remove_RT = []
        Indices_to_remove_Acc = []
        NoReaction_idx = []
        # Lists to store sensor attributed (noise and offset corrections), to be converted to dataframes later
        AccelerationNoiseList = []
        GyroNoiseList = []
        CalibratedList = []
        # Lists to keep track of how much data is present per participant, such that participants with too little
        # valid trials can be removed based on MinDataRatio cutoff. 
        ParticipantIDX = []
        PreviousParticipantName = 'DummyParticipant'

        for i, row in iterrows:
            # Get current participant name
            CurrentParticipantName = row[self.constants['PARTICIPANT_COLUMN']]
            # Check if the current row participant is different from the previous row participant
            if CurrentParticipantName != PreviousParticipantName:
                # Store row indices of the start of each unique participant ID 
                ParticipantIDX.append(i)
                PreviousParticipantName = CurrentParticipantName
            elif i == len(iterrows) - 1:
                # Store last index to get IDX range of last participant
                ParticipantIDX.append(i)

            # Remove implausible reaction times
            if RemoveRT:
                # Check if trial RT is realistic, or if there is No reaction
                _HasRealisticRT, NoReaction = self._HasRealisticRT(row, RTThreshold)
                if NoReaction:
                    NoReaction_idx.append(i)
                if not _HasRealisticRT and i not in NoReaction_idx:
                    Indices_to_remove_RT.append(i)

            # Correct accelerations for offsets and extract noise characteristics
            if CalibrateAcc:
                # Avoid calibrations if not needed (i.e. Indices are already scheduled for removal)
                if i not in Indices_to_remove_RT:
                    CalibratedData, [a_noise, g_noise] = self._CalibrateData(row)
                else:
                    CalibratedData = DF.loc[i, :].copy(deep = True)
                    a_noise, g_noise = np.nan, np.nan
                # Store calibration and sensor noise characteristics 
                CalibratedList.append(CalibratedData)
                AccelerationNoiseList.append(a_noise)
                GyroNoiseList.append(g_noise)
            
            # Remove implausible accelerations (either due to sensor errors or otherwise)
            if RemoveAcc:
                # Need to check if there is the necessary info in the given row. 
                if not self._HasRealisticAccelerations(row, MaxArmDist):
                    Indices_to_remove_Acc.append(i)
        
        # Combine the indices to remove (and skip repeated indices)
        # If user opted to keep no reaction data (e.g. for computing more sensor noise characteristics)
        if KeepNoRT:
            Indices_to_remove = np.unique(Indices_to_remove_RT + Indices_to_remove_Acc)
        else:
            Indices_to_remove = np.unique(Indices_to_remove_RT + Indices_to_remove_Acc + NoReaction_idx)

        # If accelerations are calibrated, then add necessary columns to dataframe and replace
        # existing acceleraiton columns
        if CalibrateAcc:
            DF = pd.concat(CalibratedList, axis = 1).transpose()
            DF[self.constants['ACCELEROMETER_NOISE']] = AccelerationNoiseList
            DF[self.constants['GYRO_NOISE']] = GyroNoiseList


        NumRemoveInvalid = 0
        # Convert array to list for better appending speed
        Indices_to_remove = Indices_to_remove.tolist()
        # After all the filtering, check for participants with too little valid data
        for i in range(len(ParticipantIDX) - 1):
            PIdxRange = ParticipantIDX[i+1] - ParticipantIDX[i]
            InvalidIDX = np.where((np.array(Indices_to_remove) >= ParticipantIDX[i]) & (np.array(Indices_to_remove) <= ParticipantIDX[i+1]))[0]
            if (len(InvalidIDX)/PIdxRange) >= MinDataRatio and len(InvalidIDX) > 0:
                # Store participant ID. We cannot remove them now otherwise the Indices_to_remove would not
                # correspond to the dataframe, DF, anymore. 
                RemoveIdx = np.where(DF[self.constants['PARTICIPANT_COLUMN']].str.contains(DF.loc[ParticipantIDX[i], self.constants['PARTICIPANT_COLUMN']]))[0]
                Indices_to_remove.extend(RemoveIdx.tolist())
                NumRemoveInvalid += len(RemoveIdx)
                print('[INFO] Particiapnt {} has too little data, removing...'.format(DF.loc[ParticipantIDX[i], self.constants['PARTICIPANT_COLUMN']]))

        # Remove repeated indices
        Indices_to_remove = np.unique(Indices_to_remove)

        # Print results, if user has chosen to display info
        if self.INFO:
            print('[INFO] Filtering results:')
            if RemoveRT:
                print('[INFO] \t Percentage of data with implausible (<{} ms) reaction times: {}'.format(RTThreshold, len(Indices_to_remove_RT)/N*100))
                print('[INFO] \t Percentage of data with No Reaction: {}'.format(len(NoReaction_idx)/N*100))
            if RemoveAcc:
                print('[INFO] \t Percentage of data with implausible accelerations: {}'.format(len(Indices_to_remove_Acc)/N*100))
            print('[INFO] Total percentage of data removed due to not enough valid data: {}'.format(NumRemoveInvalid/N*100))
            print('[INFO] Total percentage of data filetered (Keep No Reactions = {}): {}'.format(KeepNoRT, len(Indices_to_remove)/N*100))
        
        # Remove data from dataframe
        Removed_Data = DF[DF.index.isin(Indices_to_remove)]
        DF = DF.drop(index = Indices_to_remove)
        DF = DF.reset_index(drop = True)

        # Return filtered dataframe and removed data
        return DF, Removed_Data



    # Function which handles the importing of the AAT data. 
    # Inputs:   None - Function uses data (i.e. raw data file paths) provided in the DataImporter class 
    #                  initialization, and other definitions within this class, to import and organize
    #                 the raw data appropriately.
    # Returns:  Data - Raw imported data of all participants (in provided raw data file path) as a 
    #                  pandas dataframe.
    def ImportData(self):
        # Initialize data as a list (to be filled in with condition tables)
        # Later on, this list will be converted into a DataFrame. Since the 
        # exact number of rows is unknown, a dataframe of N x M cannot be 
        # preallocated beforehand. 
        self._Data = []
        # If the user as opted to display information
        if self.INFO:
            print('[INFO] Loading Participant Data...')
            files = tqdm(os.listdir(self.data_path))
        else:
            files = os.listdir(self.data_path)
        # For each file in the data_path 
        for f in files:
            # We are only interested in the .json files
            # NOTE: All data for one participant is contained within a single .json file
            # Therefore, 'for f in files' is equivalent as for each participant...
            if f.endswith(".json"):
                # Extract participant data
                self._participant_file = self._LoadJson(os.path.join(self.data_path, f))
                # Check if fields 'participantId' and 'condition' exist in the participant file
                if self._CheckInfo(['participantId', 'condition'], self._participant_file):
                    # Store participant data
                    self._participant_id = self._participant_file['participantId']
                    self._condition = self._participant_file['condition']
                    self._CatQCols = self._GetCategoricalQuestionCols(self._condition)
                    self._CondTable = self._LoadParticipantData()
                    # Order by participant ID, Session, Block and Trial
                    self._CondTable = self._CondTable.reset_index().set_index(['participant', 'session', 'block', 'trial'])
                    
                    # After the second block of practice trials, the condition (e.g. Pull X) switchs (in this 
                    # case, to Push X). However, this is not explicitly recorded in the Java files. This switch
                    # can also be seen through the change in the 'correct response' (in this case, from pull to 
                    # push). However, we will identify the location of the switch the column 'is practice'
                    # Find where the word 'practice' does not occur
                    exp_trial_idx = np.where(self._CondTable['is_practice'])[0]
                    # Check for break in indices which contain the 'practice' images 
                    cond_switch = np.where([(exp_trial_idx[i] + 1) != exp_trial_idx[i + 1] for i in range(len(exp_trial_idx)-1)])[0]
                    idx_switch = exp_trial_idx[cond_switch][0]
                    # Infer new instruction
                    instructions = ['push', 'pull']
                    ori_instruction = self._CondTable['condition'][0][0:4]
                    new_instruction = instructions[np.where(instructions != np.array(ori_instruction))[0][0]]
                    # Correct condition for instruction switch
                    self._CondTable.at[idx_switch:, 'condition'] = '{}{}'.format(new_instruction, self._CondTable['condition'][0][4:])
    
                    self._CondTable[self._CatQCols] = self._CondTable[self._CatQCols].astype('category')
                    self._Data.append(self._CondTable)
        # Create pandas DataFrame from data
        Data = pd.concat(self._Data, sort = True).reset_index()
        
        return Data

    
    
    # Utility function to check if all the expected information is found in a dictionary 
    # Input:    ExpectedInfoList = List of strings of the expected keys in the dictionary 
    #           GivenInfoDict = Dictionary with information, of which will be checked to 
    #                           see if all the necessary information is present
    # Output:   output = Boolean indicating if all information is present or not
    def _CheckInfo(self, ExpectedInfoList, GivenInfoDict):
        if all(info in GivenInfoDict for info in ExpectedInfoList):
            output = True
        else:
            output = False

        return output



    # Function used to extract the experiment data from the .json files
    #   Inputs: None; This function works entirely on variables defined within the class,
    #           it calls on other functions in the class which generate necessary variables.
    #           However, the necessary constants can only be generated when the class is 
    #           initiallized (i.e. __init__ function called in the main script)
    #   Outputs: Filled in ConditionTable
    def _LoadParticipantData(self):
        # Get the relevant condition table to fill in
        self._CondTable = self._GetCondTable(self._condition)

        # Loop through the keys in the participant json file. If the keys match the 
        # (assosicated Json keys in) ExpectedParticipantData, then we fill in the 
        # corresponding column in the Condition table.
        # The ExpectedParticipantData dictionary maps the jsonKeys to the columns in
        # the CondTable, which are participant dependent but session independent.
        # For example, the participant ID. 
        # Session Sscific data is not included in the ExpectedParticipantData, and 
        # is therefore handled under the else statement below. 
        # NOTE: self._participant_file is created in the above ImportData(). Essentially, 
        # it acts as the input variable to this function, without explicitly calling it. 
        # ImportData() simply loads a participant json file (through a loop) before calling 
        # this function (in the same loop). Hence, the relevant participant data is made
        # each loop available. 
        for jsonKey in self._participant_file.keys():
            if jsonKey in self.ExpectedParticipantData.keys():
                # Fill in participant info data for a given condition, which is consistent across trials 
                # (e.g. sensor type)
                self._CondTable[self.ExpectedParticipantData[jsonKey]] = self._participant_file[jsonKey]
            else:
                # For all other fields in the participant file, check for desired data. Otherwise skip
                # irrelevant keys. 
                self._SessionDict = self._participant_file[jsonKey]
                # Extract session tasks (e.g. tasks contained in Push X session, DG (demographics) session etc.)
                for task in self._SessionDict.keys():
                    # Possible tasks are AAT or answering questionaires (e.g. for DG, a task could be 'age' which indicates
                    # participant age)
                    if task == 'AAT':
                        # Start iterable at 1 (There is no block '0')
                        for block_idx, block in enumerate(self._SessionDict[task]['blocks'], 1):
                            # First entry in each block is 'null', so we ignore the first entry (hence block[1:]). 
                            for trial_idx, trial in enumerate(block[1:], 1):
                                # Create an index which contains the session name (in this case, jsonKey, which is 
                                # something like push_x_session), the block number, and the trial number.
                                idx = (jsonKey, block_idx, trial_idx)
                                # Look up idx combination in condition table, if it exists fill in relevant data
                                if idx in self._CondTable.index:
                                    # Store block-trial level data (e.g. Correct response for Push_X session, Image used etc.)
                                    for key in self.ExpectedBlockData.keys():
                                        # Try store necessary data
                                        try:
                                            self._CondTable.at[idx, self.constants[key]] = trial[self.ExpectedBlockData[key]]
                                        # If it does not exists, fill in blank.
                                        except:
                                            self._CondTable.at[idx, self.constants[key]] = np.nan
                                    # Store data which requires special handling or additional processing
                                    # Extract the image stimulus set (e.g. Happy_04 belongs to Happy set), if it exists 
                                    try:
                                        self._CondTable.at[idx, self.constants['STIMULUS_SET_COLUMN']] = self.INV_stimulus_sets[trial[self.ExpectedBlockData['STIMULUS_COLUMN']]]
                                    except:
                                        self._CondTable.at[idx, self.constants['STIMULUS_SET_COLUMN']] = np.nan

                                    # Preallocate arrays, sizes of arrays are unknown until data is imported. 
                                    acc_times = np.array([])
                                    gyro_times = np.array([])

                                    # Extract numeric AAT trial data (e.g. acceleration, time etc.)
                                    for key, value in self.ExpectedNumericData.items():
                                        # Check if expected data is present 
                                        if value in trial.keys():
                                            # Create dataframe row to hold data from .json file
                                            val = pd.Series(trial[self.ExpectedNumericData[key]])
                                            # The indices correspond to the time, in milliseconds
                                            # In the .json file, these are stored as strings so we need to 
                                            # converthem into integers
                                            val.index = val.index.astype('int')
                                            # The time values are also stored arbitrarily, so we need to 
                                            # sort the time values to obtain a time-continuous signal
                                            val = val.sort_index()
                                            # The time arrays for acceleration and gyroscopic data are not
                                            # the same. So we need to store them separately. There is also
                                            # no 'time' key in the .json file. The times are rather given 
                                            # alongside the acceleration/gyroscopic data. 
                                            if (key.startswith('acceleration')) & (len(acc_times) == 0):
                                                acc_times = val.index.values
                                                self._CondTable.at[idx, self.constants['TIME_COLUMN']] = acc_times
                                            elif (key.startswith('gyro')) & (len(gyro_times) == 0):
                                                gyro_times = val.index.values
                                                self._CondTable.at[idx, self.constants['GYRO_TIME_COLUMN']] = gyro_times
                                            # Store acceleration/gyroscopic data based on the key (e.g. acceleration_x will
                                            # be stored under the acceleration_x column)
                                            val = val.values
                                            self._CondTable.at[idx, self.constants[self.ExpectedNumericData2Constants[key]]] = val
                                        else:
                                            # If data is missing, store as empty numpy arrays. 
                                            if len(acc_times) == 0:
                                                self._CondTable.at[idx, self.constants['TIME_COLUMN']] = np.nan
                                            elif len(gyro_times) == 0:
                                                self._CondTable.at[idx, self.constants['GYRO_TIME_COLUMN']] = np.nan
                                            self._CondTable.at[idx, self.constants[self.ExpectedNumericData2Constants[key]]] = np.nan
                    # If not an AAT session, then the session is likely a questionaire type. 
                    else:
                        # _QuestionsDict is a dictionary of the questions asked in a particular session
                        # For example, in 'DG' (demographics) has keys (questions) of 'age' and 'gender'
                        self._QuestionsDict = self._SessionDict[task]
                        for Qkey in self._QuestionsDict.keys():
                            # Note: .loc sets entire row/column to a value 
                            self._CondTable.loc[jsonKey, Qkey] = self._QuestionsDict[Qkey]['answer']

        # Convert values for "correct response" from 1 and 2 to push and pull respectively 
        self._CondTable[self.constants['CORRECT_RESPONSE_COLUMN']].replace({1:'Push',2:'Pull'}, inplace=True)            
        
        return self._CondTable



    # Function which fetches the corresponding condition table, given a condition.
    #   Input: condition of interest
    #   Output: Condition table (pandas dataframe)
    def _GetCondTable(self, condition):
        # Check if the data table already exists for a given condition
        try:
            output = self.CondTablesDict['{}'.format(condition)].copy(deep = True)
        # If not, create the necessary table for the given condiion
        except AttributeError:
            self._ImportConditions(self.cond_path)
            output = self.CondTablesDict['{}'.format(condition)].copy(deep = True)

        return output



    # Function used to get the categorical question columns from the Condition Tables
    # of a given condition
    #   Input: condition of interest
    #   Output: Categorical Question Columns (list)
    def _GetCategoricalQuestionCols(self, condition):
        # Check if categorical question columns already exists for a given condition
        try:
            output = self.CategoricalQuestionColsDict['{}'.format(condition)]
        # If not, create the necessary condition columns
        except AttributeError:
            self._ImportConditions(self.cond_path)
            output = self.CategoricalQuestionColsDict['{}'.format(condition)]

        return output
    


    # Function which generates a, mostly empty, table wherein a given participant's data may be inputted,
    # for a given condition. 
    # The tables are built based on information from the "conditions.json", "sessions.json", 'tasks.json'
    # and 'stimulus_sets.json' files. 
    #   Inputs: Path to the folder containing the files outlined above
    #   Returns: None; the function creates variables which are then used by other functions, since 
    #            several of the necessary variables in later steps are generated here.
    #            The condition tables themselves are stored in a dictionary, wherein the keys are the
    #            different conditions, and the values are the associated tables. 
    #            The condition tables may be accessed via the self.ConTablesDict variable. 
    def _ImportConditions(self, cond_path):
        # Load necessary .json files
        # conditions.json contains information pertaining to the experimental conditions, along with the 
        # associated session name. In the majority of the cases, there will be two conditions; Push X and Pull X
        # with sessions Push X session and Pull X session. 
        self.conditions = self._LoadJson(os.path.join(cond_path, "conditions.json"))
        # sessions.json contains an overview of what occurs during each session, along with an additional main session
        # The main session is used to start the overall experiment. 
        # Each session contains information pertaining to the tasks during the session; e.g. Questions, Demographics,
        # actual pull/push trials etc.
        self.sessions = self._LoadJson(os.path.join(cond_path, "sessions.json"))
        # tasks.json elaborates on the tasks of sessions.json. E.g. task 'DG' (Demographics) may have fields 'age', 'gender'
        # etc. 
        self.tasks = self._LoadJson(os.path.join(cond_path, "tasks.json"))
        # stimulus_sets.json contains the names of each stimulus set, along with the names of the images used in each set
        # e.g. Stimulus Happy contians Happy_01, Happy_02, ..., Happy_NN
        self.stimulus_sets = self._LoadJson(os.path.join(cond_path, "stimulus_sets.json"))
        # Invert the stimulus set such that pictures belonging to a certain stimulus set point to the correct stimulus group
        # E.g. happy_XX picture points to the happy stimulus. 
        self.INV_stimulus_sets = dict((v, k) for k in self.stimulus_sets for v in self.stimulus_sets[k])

        # Preallocate variables 
        self.CondTablesDict = {}
        self.CategoricalQuestionColsDict = {}

        # For each condition (e.g. Pull X and Push X)
        for condition in self.conditions:
            # Extract session names from conditions.json
            session_names = self.conditions[condition]['sessions']
            # Create empty lists to store question columns, numerical question columns (e.g. time since x), and categorical question
            # columns (e.g. Disagree, Neutral, Agree)
            q_cols, numeric_q_cols, cat_q_cols = [], [], []
            # Create empty lists to store block numbers, trial numbers within each block, total trial numbers and practice trials
            sessionlst, blocks, trials, total_trials, practice = [], [], [], [], []
            # For each session (Push X, Pull X etc)
            for session in session_names:
                # Extract the session tasks (e.g. demographics)
                task_names = self.sessions[session]['tasks']
                for task in task_names:
                    # Try to get the task type (e.g. questionaire -> Demographics is a type questionaire, while something like the 
                    # experiment trials would be an 'aat' type)
                    try:
                        task_type = self.tasks[task]['type']
                    except KeyError:
                        try:
                            # For tasks with similar functionalities, their contents point to a 'parent' task which holds the 
                            # necessary information
                            task = self.tasks[task]['parent']
                            task_type = self.tasks[task]['type']
                        except KeyError:
                            task_type = None
                    if task_type == 'questionnaire':
                        # Questions is a list of questions, where each question contains an ID, text, and type
                        #   Type indicates what type of question it is; e.g. Instructional, Multiple choice, etc
                        questions = self.tasks[task]['questions']
                        for i in range(len(questions)):
                            question = questions[i]
                            # Question should be a dict-like
                            if question.keys():
                                if 'type' in question.keys():
                                    # Extract the question type (e.g. Multiple choice)
                                    question_format = question['type']['format']
                                elif 'default_type' in self.tasks[task].keys():
                                    # Extract the question type (e.g. Multiple choice)
                                    question_format = self.tasks[task]['default_type']['format']
                                else:
                                    question_format = np.nan
                            else:
                                # Extract the question text. e.g. 'How many pets do you have?'
                                question = {'text':question}
                                # Leave format field empty. 
                                question_format = np.nan

                            # If the question format is not an instruction
                            if question_format != 'instruction':
                                # Extract question identifier (e.g. 'gender'), if it exists
                                if "id" in question.keys():
                                    question_id = question['id']
                                # If there is no local identifier, then we are looking at a question series
                                # e.g. Food Neophobia Scale, hence we label the question based on the 
                                # task name and the number in which it appears.
                                else:
                                    question_id = "{}_{:0>2d}".format(task, i + 1)
                                # Keep track of the question IDs if they are of special interest
                                # e.g. numerical questions for height and weight to calculate BMI later on
                                if question_format in self.constants['NUMERICAL_QUESTION_TYPES']:
                                    numeric_q_cols.append(question_id)
                                # e.g. categorical questions to determine trait anger
                                if question_format in self.constants['CATEGORICAL_QUESTION_TYPES']:
                                    cat_q_cols.append(question_id)
                                # Append all questions to a separate column
                                q_cols.append(question_id)
                            # If question_format is an instruction, skip. 
                            else:
                                pass
                    # If participants are doing the AAT trials 
                    elif task_type == 'aat':
                        
                        # Count the number of practice images from stimulus_sets.json. 
                        # Here, targets are one of the stimulus types (e.g. happy faces) and controls are the other (e.g. angry faces)
                        num_practice = sum([len([i]) for i in self.stimulus_sets[self.tasks[task]['practice_targets'][0]]])
                        num_practice += sum([len([i]) for i in self.stimulus_sets[self.tasks[task]['practice_controls'][0]]])
                        
                        # Check if there is a specified number of repititions, if not default to 1. 
                        target_rep = max(1, float(self.tasks[task]['target_rep']))
                        # Count the number of experiment images, from one stimulus set
                        num_stim = sum([len([i]) for i in self.stimulus_sets[self.tasks[task]['targets'][0]]]) * target_rep
                        # Check if there is a specified number of repititions, if not default to 1. 
                        control_rep = max(1, float(self.tasks[task]['control_rep']))
                        # Add number of experiment images from other stimulus set to get total number of experiment images
                        num_stim += sum([len([i]) for i in self.stimulus_sets[self.tasks[task]['controls'][0]]]) * control_rep

                        # Extract the number of blocks 
                        num_blocks = self.tasks[task]['amount_of_blocks']
                        cumulative_trials = 1

                        # For each block
                        for b in range(num_blocks):
                            # Practice blocks are given by b = 0 and b = num_stim/2. The second practice blocks 
                            # corresponds to the inverted condition (e.g. Pull X --> Push X)
                            if b == 0 or b == num_blocks/2:
                                num_trials = int(num_practice/2.)
                                is_practice = True
                            else:
                                num_trials = int(num_stim/(num_blocks - 2))
                                is_practice = False

                            # For each of the experiment trials (i.e. sets of Push X and Pull Y)
                            for t in range(0, num_trials):
                                # Record session (e.g. Push X), Block #, Within-block trial #
                                # total trial #, and if current trial is a practice session. 
                                sessionlst.append(session)
                                blocks.append(b + 1)
                                trials.append(t + 1)
                                total_trials.append(cumulative_trials)
                                practice.append(is_practice)
                                # Add to total trials
                                cumulative_trials += 1
                    # If task type is not questionaire or aat (e.g. instruction, informed consent), store session
                    # only 
                    else:
                        sessionlst.append(session)
                        blocks.append(np.nan)
                        trials.append(np.nan)
                        total_trials.append(np.nan)
                        practice.append('NA')

            # Create full list of columns
            cols = self.cols + sorted(q_cols) + self.aat_cols

            # Create dataframe with necessary columns, for each condition. 
            condition_table = pd.DataFrame(columns = cols, index = list(range(len(sessionlst))))

            # Fill in columns for which data is known. 
            cols_with_data = ['CONDITION_COLUMN', 'SESSION_COLUMN', 'BLOCK_COLUMN',
                              'IS_PRACTICE_COLUMN', 'TRIAL_NUMBER_COLUMN', 'TRIAL_NUMBER_CUM_COLUMN']
            # NOTE: Order in the data list should mirror that of the "cols_with_data" variable above
            data_list = [condition, sessionlst, blocks, practice, trials, total_trials]
            for i in range(len(cols_with_data)):
                condition_table[self.constants[cols_with_data[i]]] = data_list[i]

            # Set custom index for the dataframe for easy referencing when inputting AAT data. 
            # Structure as index = [Session Column, Block Column, Trial Number]
            # Example index: ['push_happy', '2', '10'] -> Push Happy session, block 2, trial 10. 
            condition_table = condition_table.set_index([self.constants['SESSION_COLUMN'], self.constants['BLOCK_COLUMN'], self.constants['TRIAL_NUMBER_COLUMN']]).sort_index()

            # Store table under relevant condition, such that it can be accessed by other functions in the
            # DataImporter object. 
            self.CondTablesDict.update({'{}'.format(condition) : condition_table})
            self.CategoricalQuestionColsDict.update({'{}'.format(condition) : cat_q_cols})
        return None



    # General function to load Json files.
    #   Inputs: path = Full file path (incl .json)
    #   Returns: json file data
    def _LoadJson(self, path):
        # Check if file exists
        if os.path.isfile(path):
            # Open file, extract contents, then close the file
            with open(path, 'r', encoding = 'utf-8') as f:
                self.jsonfile = json.loads(f.read(), strict = False)
            f.close()
        return self.jsonfile




# Class to facilitate plotting of the AAT data
class Plotter:

    def __init__(self, constants):
        self.constants = constants


    # Function to display plots
    def ShowPlots(self):
        plt.show()
        return None


    # Function to plot the acceleration, in the x, y, and z directions, as a function of time
    # Units: Acceleration in m/s^2 and Time in ms (milliseconds)
    def AccelerationTime(self, participant, DF):
        Stimulus = DF[self.constants['STIMULUS_COLUMN']][participant]
        Correct_Response = DF[self.constants['CORRECT_RESPONSE_COLUMN']][participant]
        IsPractice = DF[self.constants['IS_PRACTICE_COLUMN']][participant]

        plt.figure()
        plt.title('Stimulus: {}, Correct Response: {}, Is Practice Trial: {}'.format(Stimulus, Correct_Response, IsPractice))
        plt.plot(DF[self.constants['TIME_COLUMN']][participant], DF[self.constants['ACCELERATION_X_COLUMN']][participant], label = 'X')
        plt.plot(DF[self.constants['TIME_COLUMN']][participant], DF[self.constants['ACCELERATION_Y_COLUMN']][participant], label = 'Y')
        plt.plot(DF[self.constants['TIME_COLUMN']][participant], DF[self.constants['ACCELERATION_COLUMN']][participant], label = 'Z')
        
        plt.legend()
        plt.grid()
        plt.xlabel('Time [ms]')
        plt.ylabel(r'Acceleration $\frac{m}{s^2}$')

        return None


    # Function to plot the displacement, in the x, y, and z directions, as a function of time
    # Units: Displacement in cm and Time in ms (milliseconds)
    def DistanceTime(self, participant, DF, Threshold = 60):
        Stimulus = DF[self.constants['STIMULUS_COLUMN']][participant]
        Correct_Response = DF[self.constants['CORRECT_RESPONSE_COLUMN']][participant]
        IsPractice = DF[self.constants['IS_PRACTICE_COLUMN']][participant]

        # Define data points which represent realistic limits of maximum distance
        Distance_UpperBound = DF[self.constants['TIME_COLUMN']][participant] * 0 + Threshold
        Distance_LowerBound = DF[self.constants['TIME_COLUMN']][participant] * 0 - Threshold

        max_dist = np.max([np.max(np.abs(DF[self.constants['DISTANCE_X_COLUMN']][participant]*100)), np.max(np.abs(DF[self.constants['DISTANCE_Y_COLUMN']][participant]*100)), np.max(np.abs(DF[self.constants['DISTANCE_Z_COLUMN']][participant]*100))])

        plt.figure()
        plt.title('Stimulus: {}, Correct Response: {}, Is Practice Trial: {}'.format(Stimulus, Correct_Response, IsPractice))
        plt.plot(DF[self.constants['TIME_COLUMN']][participant], DF[self.constants['DISTANCE_X_COLUMN']][participant]*100, label = 'X')
        plt.plot(DF[self.constants['TIME_COLUMN']][participant], DF[self.constants['DISTANCE_Y_COLUMN']][participant]*100, label = 'Y')
        plt.plot(DF[self.constants['TIME_COLUMN']][participant], DF[self.constants['DISTANCE_Z_COLUMN']][participant]*100, label = 'Z')
        
        # Plot threshold limits only if maximum distance is close to these limits
        if max_dist >= 0.8*Threshold:
            plt.plot(DF[self.constants['TIME_COLUMN']][participant], Distance_UpperBound, color = 'k', linestyle = '--', label = 'Maximum Realistic Distance')
            plt.plot(DF[self.constants['TIME_COLUMN']][participant], Distance_LowerBound, color = 'k', linestyle = '--')

        plt.legend()
        plt.grid()
        plt.xlabel('Time [ms]')
        plt.ylabel(r'Distance [cm]')
        
        return None


    # Function to plot the trajectory, in 3D, of the AAT movement. 
    # Since the 3D plot spans dimensions x, y, and z, the temporal information is not explicitly present in
    # this trajectory
    # There are two options to include the temporal information
    # By Default (Gradient = False), the script will add two colored points which correspond to the start 
    # and end of the movement. 
    # The exact mapping of color to start/stop will be displayed on the legend
    # If Gradient = True, then the temporal information will be illustrated through the line color of the 
    # trajectory. The color map used can be specified by ColorMap. Again, two points will be displayed which
    # indicate the start and end of the movement. The colors will be matched to the extremes of ColorMap
    #
    # Alternatively, AnimateTrajectory3D can be used to show the propagation of the trajectory in time. 
    def Trajectory3D(self, participant, DF, Gradient = False, ColorMap = 'RdYlGn_r'):
        Stimulus = DF[self.constants['STIMULUS_COLUMN']][participant]
        Correct_Response = DF[self.constants['CORRECT_RESPONSE_COLUMN']][participant]

        X = DF[self.constants['DISTANCE_X_COLUMN']][participant]*100
        Y = DF[self.constants['DISTANCE_Y_COLUMN']][participant]*100
        Z = DF[self.constants['DISTANCE_Z_COLUMN']][participant]*100
        time = DF[self.constants['TIME_COLUMN']][participant]

        max_val = np.max([np.max(X), np.max(Y), np.max(Z)])
        min_val = np.min([np.min(X), np.min(Y), np.min(Z)])

        if Gradient:
            points = np.array([Z, Y, X]).transpose().reshape(-1, 1, 3)
            # Map points to line segments of neighboring points (e.g. point i with point i + 1)
            segments = np.concatenate([points[:-1], points[1:]], axis = 1)
            # Make collection of line segments
            lc = Line3DCollection(segments, cmap=ColorMap)
            # Color segments based on time
            lc.set_array(time)

        plt.figure()
        ax = plt.axes(projection = '3d')
        plt.title('Stimulus: {}, Correct Response: {}'.format(Stimulus, Correct_Response))
        
        # If we want to represent time as a color gradient on the trajectory
        if Gradient:
            cmap = mpl.cm.get_cmap(ColorMap)
            ax.add_collection3d(lc)
            # Indicate where the motion began and where it ended, since temporal information is not displayed 
            ax.scatter(Z[0], Y[0], X[0], c = [cmap(0)], label = 'Start of movement (t = {} [ms])'.format(time[0]))
            ax.scatter(Z[-1], Y[-1], X[-1], c = [cmap(time[-1])], label = 'End of movement (t = {} [ms])'.format(time[-1]))
        else:
            ax.plot3D(Z, Y, X, label = 'Trajectory (displacement)')
            # Indicate where the motion began and where it ended, since temporal information is not displayed 
            ax.scatter(Z[0], Y[0], X[0], c = 'g', label = 'Start of movement (t = {} [ms])'.format(time[0]))
            ax.scatter(Z[-1], Y[-1], X[-1], c = 'r', label = 'End of movement (t = {} [ms])'.format(time[-1]))
        
        ax.legend()
        ax.set_xlim3d(min_val, max_val)
        ax.set_ylim3d(min_val, max_val)
        ax.set_zlim3d(min_val, max_val)
        ax.set_xlabel('Apporach/Avoidance (z-axis) [cm]')
        ax.set_ylabel('Lateral Movement (y-axis) [cm]')
        ax.set_zlabel('Vertical Movement (x-axis) [cm]')

        return None


    # Function to plot the acceleration, in 3D, of the AAT movement. 
    # Since the 3D plot spans dimensions x, y, and z, the temporal information is not explicitly present here
    # There are two options to include the temporal information:
    # By Default (Gradient = False), the script will add two colored points which correspond to the start 
    # and end of the movement. 
    # The exact mapping of color to start/stop will be displayed on the legend
    # If Gradient = True, then the temporal information will be illustrated through the line color of the 
    # 3D acceleration. The color map used can be specified by ColorMap. Again, two points will be displayed 
    # which indicate the start and end of the movement. The colors will be matched to the extremes of ColorMap
    #
    # Alternatively, AnimateAcceleration3D can be used to show the propagation of acceleration in time. 
    def Acceleration3D(self, participant, DF, Gradient = False, ColorMap = 'RdYlGn_r'):
        Stimulus = DF[self.constants['STIMULUS_COLUMN']][participant]
        Correct_Response = DF[self.constants['CORRECT_RESPONSE_COLUMN']][participant]

        X = DF[self.constants['ACCELERATION_X_COLUMN']][participant]
        Y = DF[self.constants['ACCELERATION_Y_COLUMN']][participant]
        Z = DF[self.constants['ACCELERATION_COLUMN']][participant]
        time = DF[self.constants['TIME_COLUMN']][participant]

        max_val = np.max([np.max(X), np.max(Y), np.max(Z)])
        min_val = np.min([np.min(X), np.min(Y), np.min(Z)])

        if Gradient:
            points = np.array([Z, Y, X]).transpose().reshape(-1, 1, 3)
            # Map points to line segments of neighboring points (e.g. point i with point i + 1)
            segments = np.concatenate([points[:-1], points[1:]], axis = 1)
            # Make collection of line segments
            lc = Line3DCollection(segments, cmap=ColorMap)
            # Color segments based on time
            lc.set_array(time)

        plt.figure()
        ax = plt.axes(projection = '3d')
        plt.title('Stimulus: {}, Correct Response: {}'.format(Stimulus, Correct_Response))
        
        # If we want to represent time as a color gradient on the trajectory
        if Gradient:
            cmap = mpl.cm.get_cmap(ColorMap)
            ax.add_collection3d(lc)
            # Indicate where the motion began and where it ended, since temporal information is not displayed 
            ax.scatter(Z[0], Y[0], X[0], c = [cmap(0)], label = 'Start of movement (t = {} [ms])'.format(time[0]))
            ax.scatter(Z[-1], Y[-1], X[-1], c = [cmap(time[-1])], label = 'End of movement (t = {} [ms])'.format(time[-1]))
        else:
            ax.plot3D(Z, Y, X, label = 'Acceleration')
            # Indicate where the motion began and where it ended, since temporal information is not displayed 
            ax.scatter(Z[0], Y[0], X[0], c = 'g', label = 'Start of movement (t = {} [ms])'.format(time[0]))
            ax.scatter(Z[-1], Y[-1], X[-1], c = 'r', label = 'End of movement (t = {} [ms])'.format(time[-1]))

        ax.legend()
        ax.set_xlim3d(min_val, max_val)
        ax.set_ylim3d(min_val, max_val)
        ax.set_zlim3d(min_val, max_val)
        ax.set_xlabel(r'Apporach/Avoidance (z-axis) $[\frac{m}{s^2}]$')
        ax.set_ylabel(r'Lateral Movement (y-axis) $[\frac{m}{s^2}]$')
        ax.set_zlabel(r'Vertical Movement (x-axis) $[\frac{m}{s^2}]$')

        return None


    # Function to show the 3D propagation of displacement in time of the AAT movement.
    # Inputs:   Participant = Index in the dataframe, DF, that we would like to plot
    #           DF = DataFrame containing the AAT data
    #           UseBlit = Boolean to use blit when rendering animation. Typically, 
    #               UseBilt = True will result in a faster and smoother animation
    #               but it does not allow free movement around the plot. To see
    #               other regions of the plot, set UseBilt = False or modify the 
    #               input 'View'
    #           View = Parameters to set the 'camera' view of the 3D plot by specifying
    #               the elevation and azimuth. Here, View = [elevation, azimuth]
    #           Save = Boolean which dictates whether the animation should be saved or not
    # Output:   An animation of the trajectory, as a function of time. 
    def AnimateTrajectory3D(self, participant, DF, UseBlit=False, View=[20, -60], Save=False):
        # Function to update data being shown in the plot
        def update(num, data, line):
            line.set_data(data[:2, :num])
            line.set_3d_properties(data[2, :num])
            return line, 

        Stimulus = DF[self.constants['STIMULUS_COLUMN']][participant]
        Correct_Response = DF[self.constants['CORRECT_RESPONSE_COLUMN']][participant]

        X = DF[self.constants['DISTANCE_X_COLUMN']][participant]*100
        Y = DF[self.constants['DISTANCE_Y_COLUMN']][participant]*100
        Z = DF[self.constants['DISTANCE_Z_COLUMN']][participant]*100
        time = DF[self.constants['TIME_COLUMN']][participant]

        max_val = np.max([np.max(X), np.max(Y), np.max(Z)])
        min_val = np.min([np.min(X), np.min(Y), np.min(Z)])

        MovFig = plt.figure()
        ax = mplot3d.axes3d.Axes3D(MovFig)

        plt.title('Stimulus: {}, Correct Response: {}'.format(Stimulus, Correct_Response))

        data3Dim = np.array([Z, Y, X])
        lines, = ax.plot(data3Dim[0, 0:1], data3Dim[1, 0:1], data3Dim[2, 0:1], label='Trajectory')

        ax.set_xlabel('Apporach/Avoidance (z-axis) [cm]')
        ax.set_ylabel('Lateral Movement (y-axis) [cm]')
        ax.set_zlabel('Vertical Movement (x-axis) [cm]')
        ax.set_xlim3d(min_val, max_val)
        ax.set_ylim3d(min_val, max_val)
        ax.set_zlim3d(min_val, max_val)

        # Set viewpoint; elev = Elevation in degrees; azim = Azimuth in degrees
        ax.view_init(elev=View[0], azim=View[1])
        
        ax.legend(loc='lower left', bbox_to_anchor = (0.7, 0.7))

        line_animation = animation.FuncAnimation(MovFig, update, len(time), fargs=(data3Dim, lines), interval=1, blit=UseBlit)
        plt.show()

        if Save:
            line_animation.save('Trajectory_{}_Stim-{}_CRes-{}.mp4'.format(participant, Stimulus, Correct_Response), writer='FFMpegWriter')

        return None

    # Function to show the 3D propagation of acceleration in time of the AAT movement.
    # Inputs:   Participant = Index in the dataframe, DF, that we would like to plot
    #           DF = DataFrame containing the AAT data
    #           UseBlit = Boolean to use blit when rendering animation. Typically, 
    #               UseBilt = True will result in a faster and smoother animation
    #               but it does not allow free movement around the plot. To see
    #               other regions of the plot, set UseBilt = False or modify the 
    #               input 'View'
    #           View = Parameters to set the 'camera' view of the 3D plot by specifying
    #               the elevation and azimuth. Here, View = [elevation, azimuth]
    #           Save = Boolean which dictates whether the animation should be saved or not
    # Output:   An animation of the 3D acceleration, as a function of time. 
    def AnimateAcceleration3D(self, participant, DF, UseBlit=False, View=[20, -60], Save=False):
        # Function to update data being shown in the plot
        def update(num, data, line):
            line.set_data(data[:2, :num])
            line.set_3d_properties(data[2, :num])
            return line, 

        Stimulus = DF[self.constants['STIMULUS_COLUMN']][participant]
        Correct_Response = DF[self.constants['CORRECT_RESPONSE_COLUMN']][participant]

        X = DF[self.constants['ACCELERATION_X_COLUMN']][participant]
        Y = DF[self.constants['ACCELERATION_Y_COLUMN']][participant]
        Z = DF[self.constants['ACCELERATION_COLUMN']][participant]
        time = DF[self.constants['TIME_COLUMN']][participant]


        max_val = np.max([np.max(X), np.max(Y), np.max(Z)])
        min_val = np.min([np.min(X), np.min(Y), np.min(Z)])

        MovFig = plt.figure()
        ax = mplot3d.axes3d.Axes3D(MovFig)

        plt.title('Stimulus: {}, Correct Response: {}'.format(Stimulus, Correct_Response))

        data3Dim = np.array([Z, Y, X])
        lines, = ax.plot(data3Dim[0, 0:1], data3Dim[1, 0:1], data3Dim[2, 0:1], label = 'Acceleration')

        ax.set_xlabel(r'Apporach/Avoidance (z-axis) $[\frac{m}{s^2}]$')
        ax.set_ylabel(r'Lateral Movement (y-axis) $[\frac{m}{s^2}]$')
        ax.set_zlabel(r'Vertical Movement (x-axis) $[\frac{m}{s^2}]$')
        ax.set_xlim3d(min_val, max_val)
        ax.set_ylim3d(min_val, max_val)
        ax.set_zlim3d(min_val, max_val)

        # Set viewpoint; elev = Elevation in degrees; azim = Azimuth in degrees
        ax.view_init(elev=View[0], azim=View[1])
        
        ax.legend(loc='lower left', bbox_to_anchor = (0.7, 0.7))

        line_animation = animation.FuncAnimation(MovFig, update, len(time), fargs=(data3Dim, lines), interval=1, blit=UseBlit)
        plt.show()

        if Save:
            line_animation.save('3DAcceleration_{}_Stim-{}_CRes-{}.mp4'.format(participant, Stimulus, Correct_Response), writer='FFMpegWriter')

        return None