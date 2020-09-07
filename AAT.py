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
import PySimpleGUI as sg
import time

from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt
from matplotlib import animation as animation
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib as mpl
import statsmodels.api as sm
import statsmodels.formula.api as smf

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
                        'TOTAL_DISTANCE_COLUMN' : 'total_distance',
                        'DELTA_A_COLUMN': 'da',
                        'DELTA_D_COLUMN': 'dd',
                        '_globalTrials': '_globalTrials'}
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

        self.CondensedCols = [self.constants['PARTICIPANT_COLUMN'],
                                    self.constants['STIMULUS_SET_COLUMN'],
                                    self.constants['CORRECT_RESPONSE_COLUMN'],
                                    self.constants['STIMULUS_COLUMN'],
                                    self.constants['CONDITION_COLUMN'],
                                    self.constants['SESSION_COLUMN'],
                                    self.constants['BLOCK_COLUMN'],
                                    self.constants['TRIAL_NUMBER_CUM_COLUMN']]


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
        if filename[-3:] == 'csv':
            dataframe.to_csv('./{}'.format(filename))
        else:
            dataframe.to_pickle('./{}'.format(filename))
        os.chdir(self.cwd)
        return None



    # Utility function to load saved dataframes
    # Input:    filename = The name of the file to be loaded
    #           path = Path to the file
    # Output:   df = DataFrame containing data from filename
    def LoadDF(self, filename, path):
        if path is not None:
            os.chdir(path)
            df = pd.read_pickle(os.path.join(path, filename))
            os.chdir(self.cwd)
        else:
            df = pd.read_pickle(filename)
        return df



    def CondenseDF(self, DataFrame, AddQuestions = True, sgParams = None):
        DF = DataFrame.copy(deep = True)
        if AddQuestions:
            for key in DataFrame.columns:
                if key not in self.constants.values() and not key.startswith('DG') and key != 'completion':
                    self.CondensedCols.append(key)
        if all(col in DF.columns for col in self.CondensedCols):
            try:
                Mapper = {self.constants['RT_COLUMN']:"Reaction Time"}
                self.CondensedCols.append(self.constants['RT_COLUMN'])
                if self.constants['DELTA_A_COLUMN'] in DF.columns:
                    Mapper.update({self.constants['DELTA_A_COLUMN']:"Reaction Force"})
                    self.CondensedCols.append(self.constants['DELTA_A_COLUMN'])
                if self.constants['DELTA_D_COLUMN'] in DF.columns:
                    Mapper.update({self.constants['DELTA_D_COLUMN']:"Reaction Distance"})
                    self.CondensedCols.append(self.constants['DELTA_D_COLUMN'])
                DF = DF[self.CondensedCols]
                # Rename the reaction time, reaction force and reaction distance columns to be
                # for those using the GUI, such that the output data is easier to interpret. 
                if sgParams is not None:
                    DF = DF.rename(columns = Mapper)
            except KeyError:
                if sgParams is not None:
                    sgParams['OUTPUT'].print('[ WARNING ] Cannot condense columns since the Reaction Time column is missing.', text_color = 'red')
                if self.INFO:
                    print('[ WARNING ] Cannot condense columns since the expected columns do not exist.')
        else:
            if self.INFO:
                print('[ WARNING ] Cannot condense columns since the expected columns do not exist.')
            if sgParams is not None:
                sgParams['OUTPUT'].print('[ WARNING ] Cannot condense columns since the expected columns do not exist. Defaulting to standard DF.', text_color = 'red')
        return DF

    
    # Function to compute the peak accelerations and distances following a reaction
    # Input:    processedData = DataFrame containing the processed AAT data (i.e.
    #                           with the computed distances)
    #           axes = Axes for which the peak acceleration and distance will be 
    #                  computed. If multiple axes are provided, then the function
    #                  will compute the magnitude of the peaks in the provided 
    #                  axes. The default is the Z (approach/avoidance) axis. 
    #           absolute = Boolean indicating if the absolute value of the 
    #                      peak accelerations and distances should be taken
    #                      Default = False
    # Output:   DF: Updated version of the input 'processedData' containing
    #               the peak accelerations and distances
    def ComputeDeltaAandD(self, processedData, axes = ['Z'], absolute=False, which = ['acceleration', 'distance'], sgParams=None):

        def _Delta(row, axes, Map, pos = 1):
            t = row[self.constants['TIME_COLUMN']]
            rt = row[self.constants['RT_COLUMN']]
            x = np.zeros((len(axes), len(t)))

            for i, ax in enumerate(axes):
                x[i, :] = row[self.constants[Map[ax][pos]]]

            if i > 0:
                x = np.sqrt(np.nansum(np.square(x), axis = 0))

            x = x.reshape((len(row[self.constants['TIME_COLUMN']]), ))

            try:
                idxs = spsig.find_peaks(abs(x))[0]
                idx = np.where(t[idxs] >= rt)[0][0]
                dx = x[idxs[idx]]
                if absolute:
                    dx = abs(dx)
            except IndexError:
                dx = np.nan

            return dx

        DF = processedData.copy(deep = True)

        AxisMap = {'X':['DISTANCE_X_COLUMN', 'ACCELERATION_X_COLUMN'], 
            'Y':['DISTANCE_Y_COLUMN', 'ACCELERATION_Y_COLUMN'], 
            'Z':['DISTANCE_Z_COLUMN', 'ACCELERATION_COLUMN']}

        if 'acceleration' in which:
            if self.INFO:
                print('[INFO] Computing reaction force (RF)...')
            if sgParams:
                sgParams['OUTPUT'].print('[ INFO ] Computing reaction force (RF)...')
                sgParams['PERCENTAGE'].Update('- %')
                sgParams['TIME'].Update('Time remaining: - [s]')
            DF[self.constants['DELTA_A_COLUMN']] = DF.apply(lambda row: _Delta(row, axes, AxisMap, pos = 1), axis = 1)
        if 'distance' in which:
            if self.INFO:
                print('[INFO] Computing reaction distance (RD)...')
            if sgParams:
                sgParams['OUTPUT'].print('[ INFO ] Computing reaction distance (RD)...')
                sgParams['PERCENTAGE'].Update('- %')
                sgParams['TIME'].Update('Time remaining: - [s]')
            DF[self.constants['DELTA_D_COLUMN']] = DF.apply(lambda row: _Delta(row, axes, AxisMap, pos = 0), axis = 1)

        return DF



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
    def ComputeDistance(self, DataFrame, ComputeTotalDistance = False, sgParams = None):

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

        if sgParams:
            sgParams['OUTPUT'].print('[ INFO ] Computing distances...')

        DF_out_list = []

        execution_time = 0
        sTime = time.time()

        for idx, row in iterrows:
            if sgParams:
                # # Update progress bar, if user hits cancel, then break the iteration loop
                # if not sg.OneLineProgressMeter('Filtering data...', i + 1, len(files), key='-IMPORT_PROGRESS-', orientation='h'):
                #     break
                sgParams['PERCENTAGE'].Update('{:2d} %'.format(int(100*(idx+1)/DF.shape[0])))
                sgParams['PROG_BAR'].UpdateBar(idx + 1, DF.shape[0])
                timeRemainingS = execution_time/(idx + 1) * (DF.shape[0] - idx + 1)
                sgParams['TIME'].Update('Time remaining:{:6d} [s]'.format(int(timeRemainingS)))
                events, values = sgParams['WINDOW'].read(timeout=1)
                if events in (sg.WIN_CLOSED, '-CANCEL-'):
                    sgParams.update({'IS_CLOSED':True})
                    break
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

            execution_time = time.time() - sTime

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
    def Correct4Rotations(self, DataFrame, StoreTheta = True, sgParams = None):
        DF = DataFrame.copy(deep = True)

        # If the user decided to display information to the terminal window
        if self.INFO:
            print("[INFO] Correction for rotations...")
            iterrows = tqdm(DF.iterrows(), total=DF.shape[0])
        else:
            iterrows = DF.iterrows()
        
        if sgParams:
            sgParams['OUTPUT'].print('[ INFO ] Correcting accelerations for (any) rotations during reaction...')
        
        # Preallocate list to store corrected data, which will later be compiled
        # into the output dataframe
        CorrectedDataList = []

        # Define intial angle (assume 0 since device is stationary)
        # Even if the device is tilted, setting the intial angle to zero will
        # give the change in angle relative to this position
        InitialTheta = np.zeros((3, 1))
        # Set angle threshold, above which a rotation is considered to be occuring
        AngleThreshold = 1*np.pi/180.

        execution_time = 0
        sTime = time.time()

        for idx, row in iterrows:
            if sgParams:
                # # Update progress bar, if user hits cancel, then break the iteration loop
                # if not sg.OneLineProgressMeter('Filtering data...', i + 1, len(files), key='-IMPORT_PROGRESS-', orientation='h'):
                #     break
                sgParams['PERCENTAGE'].Update('{:2d} %'.format(int(100*(idx+1)/DF.shape[0])))
                sgParams['PROG_BAR'].UpdateBar(idx + 1, DF.shape[0])
                timeRemainingS = execution_time/(idx + 1) * (DF.shape[0] - idx + 1)
                sgParams['TIME'].Update('Time remaining:{:6d} [s]'.format(int(timeRemainingS)))
                events, values = sgParams['WINDOW'].read(timeout=1)
                if events in (sg.WIN_CLOSED, '-CANCEL-'):
                    sgParams.update({'IS_CLOSED':True})
                    break
        
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
            execution_time = time.time() - sTime

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
    def ResampleData(self, DataFrame, sgParams = None):
        DF = DataFrame.copy(deep = True)
        
        # If the user decided to display information to the terminal window
        if self.INFO:
            print("[INFO] Running Resampling...")
            iterrows = tqdm(DF.iterrows(), total=DF.shape[0])
        else:
            iterrows = DF.iterrows()
        if sgParams:
            sgParams['OUTPUT'].print('[ INFO ] Resampling the sensor data...')

        ResampledList = []

        execution_time = 0
        sTime = time.time()

        # Resample the rows. The columns specified by DataCols2Resample in __init__()
        # will be resampled
        for i, row in iterrows:
            if sgParams:
                # # Update progress bar, if user hits cancel, then break the iteration loop
                # if not sg.OneLineProgressMeter('Filtering data...', i + 1, len(files), key='-IMPORT_PROGRESS-', orientation='h'):
                #     break
                sgParams['PERCENTAGE'].Update('{:2d} %'.format(int(100*(i+1)/DF.shape[0])))
                sgParams['PROG_BAR'].UpdateBar(i + 1, DF.shape[0])
                timeRemainingS = execution_time/(i + 1) * (DF.shape[0] - i + 1)
                sgParams['TIME'].Update('Time remaining:{:6d} [s]'.format(int(timeRemainingS)))
                events, values = sgParams['WINDOW'].read(timeout=1)
                if events in (sg.WIN_CLOSED, '-CANCEL-'):
                    sgParams.update({'IS_CLOSED':True})
                    break

            _ResampledRow = self._ResampleRow(row)
            ResampledList.append(_ResampledRow)

            execution_time = time.time() - sTime

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
    def FilterNaNs(self, DataFrame, sgParams = None):
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
        
        if sgParams:
            sgParams['OUTPUT'].print('[ INFO ] Removing missing data...')
            sgParams['OUTPUT'].print('[ INFO ]\t Percentage of data which is missing: {:3d} %'.format(int(len(Indices_to_remove_NaN)/N*100)), text_color = 'blue')


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
    def ComputeRT(self, DataFrame, RTResLB = 0.8, RTHeightThresRatio = 0.3, RTPeakDistance = 10, sgParams = None):
        DF = DataFrame.copy(deep = True)
        # If the user decided to display information to the terminal window
        if self.INFO:
            print("[INFO] Computing Reaction Times...")
            iterrows = tqdm(DF.iterrows(), total=DF.shape[0])
        else:
            iterrows = DF.iterrows()
        if sgParams:
            sgParams['OUTPUT'].print('[ INFO ] Computing Reaction times...')

        # Define constants and preallocate empty Dataframe series to host RTs
        N = DF.shape[0]
        RTs = pd.Series(index = np.arange(0, N), name = self.constants['RT_COLUMN'])
        execution_time = 0
        sTime = time.time()
        for i, row in iterrows:
            if sgParams:
                # # Update progress bar, if user hits cancel, then break the iteration loop
                # if not sg.OneLineProgressMeter('Computing reaction times...', i + 1, len(files), key='-IMPORT_PROGRESS-', orientation='h'):
                #     break
                sgParams['PERCENTAGE'].Update('{:2d} %'.format(int(100*(i+1)/DF.shape[0])))
                sgParams['PROG_BAR'].UpdateBar(i + 1, DF.shape[0])
                timeRemainingS = execution_time/(i + 1) * (DF.shape[0] - i + 1)
                sgParams['TIME'].Update('Time remaining:{:6d} [s]'.format(int(timeRemainingS)))
                events, values = sgParams['WINDOW'].read(timeout=1)
                if events in (sg.WIN_CLOSED, '-CANCEL-'):
                    sgParams.update({'IS_CLOSED':True})
                    break
            # Some signals have a lot of jitter at the beginning with relatively high accelerations. To reduce the number of participants
            # falsely identified as having too fast reaction times, we will consider any peak larger than RTHeighThresRatio of the highest 
            # peak as a significant peak. However, there are also participants with no reactions, in which case we take the absolute 
            # lowerbound, defined by RTResLB. 
            RTHeightThres = max(max(abs(row[self.constants['ACCELERATION_COLUMN']]))*RTHeightThresRatio, RTResLB)
            # RTHeightThres = RTResLB
            # The try-except catch below is to account for no reactions (which may not exceed the reaction threshold)
            try:
                # Find the first index where the acceleration exceeds this threshold
                RT_idx = np.where(abs(row[self.constants['ACCELERATION_COLUMN']]) >= RTHeightThres)[0][0]
                # Find the corresponding index in the time array
                RTs[i] = row[self.constants['TIME_COLUMN']][RT_idx] - RTPeakDistance
            except IndexError:
                RTs[i] = 'No Reaction'

            # # First find all peaks in abs(signal)
            # all_peaks = spsig.find_peaks(abs(row[self.constants['ACCELERATION_COLUMN']]), distance = RTPeakDistance, height = RTHeightThres)
            # # If there are peaks, given the threshold conditions
            # if len(all_peaks[0]):
            #     # Use these all peaks to find the peak prominences: We expect that the main response will have the highest prominence values
            #     prominences = spsig.peak_prominences(abs(row[self.constants['ACCELERATION_COLUMN']]), all_peaks[0])[0]
            #     # Given the shape of the signal, there should be 2 peaks (one corresponding to the initial direction, and the other returning to 
            #     # 'origin'). The return acceleration could be higher than the initial reaction. Hence, we will take half of the highest
            #     # prominence as our threshold. 

            #     min_prominence = np.sort(prominences[::-1])[0]/2
            #     peaks = spsig.find_peaks(abs(row[self.constants['ACCELERATION_COLUMN']]), prominence = min_prominence, distance = RTPeakDistance, height = RTHeightThres)
            #     initial_peak_idx = peaks[0][0]

            #     # The final reaction time is given by the time of the peak minus the time at the start of the measurement minus the minimum peak 
            #     # distance to account for the fact that the reaction occurs before the peak. 
            #     RTs[i] = row[self.constants['TIME_COLUMN']][initial_peak_idx] - row[self.constants['TIME_COLUMN']][0] - RTPeakDistance
            # else:
            #     RTs[i] = 'No Reaction'

            execution_time = time.time() - sTime
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


    
    def _HasCorrectResponse(self, DataRow):
        Map = {'Pull':1, 'Push':-1}
        Acc = DataRow[self.constants['ACCELERATION_COLUMN']]
        rt = DataRow[self.constants['RT_COLUMN']]
        try:
            rt_idx = np.where(DataRow[self.constants['TIME_COLUMN']] > rt)[0][0]
            # Logic: If sign is positive, then the response was correct.
            # If pull then Acc is positive, so positive x positive = positive
            # If push then Acc is negative, so negative x negative = positive
            # Positive x Negative = Negative 
            cond = Acc[rt_idx] * Map[DataRow[self.constants['CORRECT_RESPONSE_COLUMN']]] > 0
        except TypeError:
            cond = False
        return cond



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
        try:
            MaxDist = MaxArmDist[DataRow['gender']]
        except KeyError:
            # Assume worst case; males have longer arms in general
            MaxDist = MaxArmDist['male']
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
    def FilterData(self, Data, RemoveIncorrectResponses = True, MinDataRatio = 0.8, RemoveRT = True, KeepNoRT = False, RTThreshold = 200, CalibrateAcc = True, RemoveAcc = True, MaxArmDist = {'female':0.735, 'male':0.810}, sgParams = None):
        # Make a copy of the data frame (such that we do not make changes to the input data frame directly)
        DF = Data.copy(deep = True)

        # Check if reaction times have been previously computed. Otherwise, compute the reaction times 
        try:
            if Data['rt'][0] > 0:
                pass
        except KeyError:
            print('[WARNING] The inputted DataFrame for <FilterData> does not have a reaction time column.\n\tI will try to compute reaction times here.\n\tNOTE: Default values are being used.')
            
            if sgParams:
                sgParams['OUTPUT'].print('[ WARNING ] The passed DataFrame does not have a reaction time column \n\tI will try to compute reaction times here.\n\tNOTE: Default values are being used.', color = 'red')
            
            DF = self.ComputeRT(DF)

        # If the user as opted to display information
        if self.INFO:
            print("[INFO] Running Filtering...")
            iterrows = tqdm(DF.iterrows(), total=DF.shape[0])
        else:
            iterrows = DF.iterrows()
        if sgParams:
            sgParams['OUTPUT'].print('[ INFO ] Filtering data...')

        # Initialize lists and constants
        N = DF.shape[0]
        # Lists to store the indices, of DF, which contain data which should be removed. 
        Indices_to_remove_RT = []
        Indices_to_remove_Acc = []
        NoReaction_idx = []
        Indices_to_remove_IncorrectRes = []
        # Lists to store sensor attributed (noise and offset corrections), to be converted to dataframes later
        AccelerationNoiseList = []
        GyroNoiseList = []
        CalibratedList = []
        # Lists to keep track of how much data is present per participant, such that participants with too little
        # valid trials can be removed based on MinDataRatio cutoff. 
        ParticipantIDX = []
        PreviousParticipantName = 'DummyParticipant'

        execution_time = 0
        sTime = time.time()

        for i, row in iterrows:
            if sgParams:
                # # Update progress bar, if user hits cancel, then break the iteration loop
                # if not sg.OneLineProgressMeter('Filtering data...', i + 1, len(files), key='-IMPORT_PROGRESS-', orientation='h'):
                #     break
                sgParams['PERCENTAGE'].Update('{:2d} %'.format(int(100*(i+1)/DF.shape[0])))
                sgParams['PROG_BAR'].UpdateBar(i + 1, DF.shape[0])
                timeRemainingS = execution_time/(i + 1) * (DF.shape[0] - i + 1)
                sgParams['TIME'].Update('Time remaining:{:6d} [s]'.format(int(timeRemainingS)))
                events, values = sgParams['WINDOW'].read(timeout=1)
                if events in (sg.WIN_CLOSED, '-CANCEL-'):
                    sgParams.update({'IS_CLOSED':True})
                    break
            # Get current participant name
            CurrentParticipantName = row[self.constants['PARTICIPANT_COLUMN']]
            # Check if the current row participant is different from the previous row participant
            if CurrentParticipantName != PreviousParticipantName:
                # Store row indices of the start of each unique participant ID 
                ParticipantIDX.append(i)
                PreviousParticipantName = CurrentParticipantName
            elif i == DF.shape[0] - 1:
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

            # Remove participants with incorrect reactions
            if RemoveIncorrectResponses:
                # Check if the response is correct or not
                IsCorrect = self._HasCorrectResponse(row)
                if not IsCorrect and i not in Indices_to_remove_RT:
                    Indices_to_remove_IncorrectRes.append(i)

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

            execution_time = time.time() - sTime

        # Combine the indices to remove (and skip repeated indices)
        # If user opted to keep no reaction data (e.g. for computing more sensor noise characteristics)
        if KeepNoRT:
            Indices_to_remove = np.unique(Indices_to_remove_RT + Indices_to_remove_Acc + Indices_to_remove_IncorrectRes)
        else:
            Indices_to_remove = np.unique(Indices_to_remove_RT + Indices_to_remove_Acc + Indices_to_remove_IncorrectRes + NoReaction_idx)

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
            if (len(InvalidIDX)/PIdxRange) >= (1 - MinDataRatio) and len(InvalidIDX) > 0:
                # Store participant ID. We cannot remove them now otherwise the Indices_to_remove would not
                # correspond to the dataframe, DF, anymore. 
                RemoveIdx = np.where(DF[self.constants['PARTICIPANT_COLUMN']].str.contains(DF.loc[ParticipantIDX[i], self.constants['PARTICIPANT_COLUMN']]))[0]
                Indices_to_remove.extend(RemoveIdx.tolist())
                NumRemoveInvalid += len(RemoveIdx)
                if self.INFO:
                    print('[INFO] Particiapnt {} has too little data, removing...'.format(DF.loc[ParticipantIDX[i], self.constants['PARTICIPANT_COLUMN']]))
                if sgParams:
                    sgParams['OUTPUT'].print('[ INFO ] Particiapnt {} has too little data, removing...'.format(DF.loc[ParticipantIDX[i], self.constants['PARTICIPANT_COLUMN']]))

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
            if RemoveIncorrectResponses:
                print('[INFO] \t Percentage of data with incorrect responses: {}'.format(len(Indices_to_remove_IncorrectRes)/N*100))
            print('[INFO] Total percentage of data removed due to not enough valid data: {}'.format(NumRemoveInvalid/N*100))
            print('[INFO] Total percentage of data filetered (Keep No Reactions = {}): {}'.format(KeepNoRT, len(Indices_to_remove)/N*100))

        if sgParams:
            sgParams['OUTPUT'].print('[ INFO ] Filtering results:')
            if RemoveRT:
                sgParams['OUTPUT'].print('[ INFO ] \t Percentage of data with implausible (<{} ms) reaction times: {:3f}'.format(RTThreshold, float(len(Indices_to_remove_RT)/N*100)), text_color = 'blue')
                sgParams['OUTPUT'].print('[ INFO ] \t Percentage of data with No Reaction: {}'.format(float(len(NoReaction_idx)/N*100)), text_color = 'blue')
            if RemoveAcc:
                sgParams['OUTPUT'].print('[ INFO ] \t Percentage of data with implausible accelerations: {:3f}'.format(float(len(Indices_to_remove_Acc)/N*100)), text_color = 'blue')
            if RemoveIncorrectResponses:
                sgParams['OUTPUT'].print('[ INFO ] \t Percentage of data with incorrect responses: {}'.format(float(len(Indices_to_remove_IncorrectRes)/N*100)), text_color = 'blue')
            sgParams['OUTPUT'].print('[ INFO ] \t Total percentage of data removed due to not enough valid data: {:3f}'.format(float(NumRemoveInvalid/N*100)), text_color = 'blue')
            sgParams['OUTPUT'].print('[ INFO ] \t Total percentage of data filetered (Keep No Reactions = {}): {:3f}'.format(KeepNoRT, float(len(Indices_to_remove)/N*100)), text_color = 'blue')
 
        
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
    def ImportData(self, sgParams = None):
        # Initialize data as a list (to be filled in with condition tables)
        # Later on, this list will be converted into a DataFrame. Since the 
        # exact number of rows is unknown, a dataframe of N x M cannot be 
        # preallocated beforehand. 
        self._Data = []
        self._DG_Cols = []
        # If the user as opted to display information
        if self.INFO:
            print('[INFO] Loading Participant Data...')
            files = tqdm(os.listdir(self.data_path))
        else:
            files = os.listdir(self.data_path)
        if sgParams:
            sgParams['OUTPUT'].print('[ INFO ] Loading Participant Data...')
        # For each file in the data_path 
        execution_time = 0
        sTime = time.time()
        for i, f in enumerate(files):
            if sgParams:
                # # Update progress bar, if user hits cancel, then break the iteration loop
                # if not sg.OneLineProgressMeter('Importing Data...', i + 1, len(files), key='-IMPORT_PROGRESS-', orientation='h'):
                #     break
                sgParams['PERCENTAGE'].Update('{:2d} %'.format(int(100*(i+1)/len(files))))
                sgParams['PROG_BAR'].UpdateBar(i + 1, len(files))
                timeRemainingS = execution_time/(i + 1) * (len(files) - i + 1)
                sgParams['TIME'].Update('Time remaining:{:6d} [s]'.format(int(timeRemainingS)))
                events, values = sgParams['WINDOW'].read(timeout=1)
                if events in (sg.WIN_CLOSED, '-CANCEL-'):
                    sgParams.update({'IS_CLOSED':True})
                    break

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
                    
                    # # After the second block of practice trials, the condition (e.g. Pull X) switchs (in this 
                    # # case, to Push X). However, this is not explicitly recorded in the Java files. This switch
                    # # can also be seen through the change in the 'correct response' (in this case, from pull to 
                    # # push). However, we will identify the location of the switch the column 'is practice'
                    # # Find where the word 'practice' does not occur
                    # exp_trial_idx = np.where(self._CondTable['is_practice'])[0]
                    # # Check for break in indices which contain the 'practice' images 
                    # cond_switch = np.where([(exp_trial_idx[i] + 1) != exp_trial_idx[i + 1] for i in range(len(exp_trial_idx)-1)])[0]
                    # idx_switch = exp_trial_idx[cond_switch + 1][0]
                    # # Infer new instruction
                    # instructions = ['push', 'pull']
                    # ori_instruction = self._CondTable['condition'][0][0:4]
                    # new_instruction = instructions[np.where(instructions != np.array(ori_instruction))[0][0]]
                    # # Correct condition for instruction switch
                    # self._CondTable.at[idx_switch:, 'condition'] = '{}{}'.format(new_instruction, self._CondTable['condition'][0][4:])

                    self._CondTable[self._CatQCols] = self._CondTable[self._CatQCols].astype('category')
                    # Rename question_id to question_label
                    for jsonKey in self._qkeys:
                        questionFields = self._participant_file[jsonKey]
                        for qf in questionFields.keys():
                            questions = self.tasks[qf]['questions']
                            for q in questions:
                                try:
                                    self._CondTable.rename(columns = {q['id']:q['label']}, inplace=True)
                                    self._DG_Cols.append(q['id'])
                                except KeyError:
                                    pass


                    self._Data.append(self._CondTable)
            execution_time = time.time() - sTime
        # Create pandas DataFrame from data
        Data = pd.concat(self._Data, sort = True).reset_index()
        try:
            Data.drop(columns=np.unique(self._DG_Cols), inplace=True)
        except KeyError:
            pass
        
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
        self._qkeys = []
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
                                if self._hasBlocksJson:
                                    for Stimkey in self.globalSwitchTrials[self._condition][jsonKey].keys():
                                        glob_trial_idx = self.globalSwitchTrials[self._condition][jsonKey][Stimkey]
                                        idx = (jsonKey, block_idx, trial_idx, glob_trial_idx + trial_idx - 1)
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
                                            
                                else:
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
                        self._qkeys.append(jsonKey)
                        for Qkey in self._QuestionsDict.keys():
                            # Note: .loc sets entire row/column to a value
                            if self._hasBlocksJson:
                                self._CondTable.loc[:, Qkey] = self._QuestionsDict[Qkey]['answer']
                            else:
                                self._CondTable.loc[jsonKey, Qkey] =  self._QuestionsDict[Qkey]['answer']


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
        # blocks.json contains a summary of the blocks used in the AAT paradigm
        try:
            self._blocks = self._LoadJson(os.path.join(cond_path, "blocks.json"))
            self._hasBlocksJson = True
            self.globalSwitchTrials = {}
        except AttributeError:
            self._hasBlocksJson = False
            pass
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
            if self._hasBlocksJson:
                self.globalTrialNum = []
                self.conditionSwitchTrials = {}
                globalCumulativeTrials = 1
            # For each session (Push X, Pull X etc)
            for session in session_names:
                # Extract the session tasks (e.g. demographics)
                task_names = self.sessions[session]['tasks']
                if self._hasBlocksJson:
                    _OldBlockID = None
                    sessionSwitchTrials = {}
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
                        # Check if aat task points to blocks in blocks.json or in tasks.
                        if self._hasBlocksJson:
                            cumulative_trials = 1
                            for block_num, block_id in enumerate(self.tasks[task]['blocks']):
                                block = self._blocks[block_id]
                                is_practice = False
                                # Check if it is a practice trial
                                if 'give_feedback' in block.keys():
                                    is_practice = block['give_feedback']
                                amount_of_trials = 0
                                for response in ['push', 'pull']:
                                    response_definition = block[response]
                                    for chooser in response_definition['stimuli']:
                                        stim_set = chooser['from']
                                        num_repetitions = 1
                                        if 'repeat' in chooser.keys():
                                            num_repetitions = chooser['repeat']
                                        amount_of_trials += chooser['pick'] * num_repetitions
                                        # If mulitple AATs are under one 'session', identify where they change by looking
                                        # at which stimset they come from. 
                                        if block_id != _OldBlockID:
                                            sessionSwitchTrials.update({block_id:globalCumulativeTrials})
                                            _OldBlockID = block_id
                                for t in range(0, int(amount_of_trials)):
                                    sessionlst.append(session)
                                    self.globalTrialNum.append(globalCumulativeTrials)
                                    blocks.append(block_num + 1)
                                    trials.append(t + 1)
                                    total_trials.append(cumulative_trials)
                                    practice.append(is_practice)
                                    # Add to total trials
                                    cumulative_trials += 1
                                    globalCumulativeTrials += 1

                        else:
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
                self.conditionSwitchTrials.update({session:sessionSwitchTrials})
            # import code
            # code.interact(local=locals())
            self.globalSwitchTrials.update({condition:self.conditionSwitchTrials})

            # Create full list of columns
            cols = self.cols + sorted(q_cols) + self.aat_cols

            # Create dataframe with necessary columns, for each condition. 
            condition_table = pd.DataFrame(columns = cols, index = list(range(len(sessionlst))))

            # Fill in columns for which data is known. 
            cols_with_data = ['CONDITION_COLUMN', 'SESSION_COLUMN', 'BLOCK_COLUMN',
                              'IS_PRACTICE_COLUMN', 'TRIAL_NUMBER_COLUMN', 'TRIAL_NUMBER_CUM_COLUMN']
            # NOTE: Order in the data list should mirror that of the "cols_with_data" variable above
            data_list = [condition, sessionlst, blocks, practice, trials, total_trials]
            if self._hasBlocksJson:
                cols_with_data.append('_globalTrials')
                data_list.append(self.globalTrialNum)
            for i in range(len(cols_with_data)):
                condition_table[self.constants[cols_with_data[i]]] = data_list[i]

            # Set custom index for the dataframe for easy referencing when inputting AAT data. 
            # Structure as index = [Session Column, Block Column, Trial Number]
            # Example index: ['push_happy', '2', '10'] -> Push Happy session, block 2, trial 10. 
            if self._hasBlocksJson:
                condition_table = condition_table.set_index([self.constants['SESSION_COLUMN'], self.constants['BLOCK_COLUMN'], self.constants['TRIAL_NUMBER_COLUMN'], self.constants['_globalTrials']]).sort_index()
            else:
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

    def __init__(self, constants = None):
        # If no constants are given, take them from the DataImporter Class
        if constants is None:
            Importer = DataImporter('DummyPath', 'DummyPath')
            self.constants = Importer.constants
        else:
            self.constants = constants



    def setFontSize(self, fontsize):
        plt.rcParams.update({'font.size':fontsize})
        return None


    # Function to display plots
    def ShowPlots(self):
        plt.show()
        return None


    # Function to plot the acceleration, in the x, y, and z directions, as a function of time
    # Units: Acceleration in m/s^2 and Time in ms (milliseconds)
    def AccelerationTime(self, participant, DF, axis = ['X', 'Y', 'Z'], movement = ['Pull', 'Push'], stimulus = None, ParentFig = None, ShowAxis = [True, True], XLims = None, YLims = None, HideLegend = False, **kwargs):
        AxisMap = {'X':['ACCELERATION_X_COLUMN', 0], 
                   'Y':['ACCELERATION_Y_COLUMN', 1], 
                   'Z':['ACCELERATION_COLUMN', 2]}

        # Necessary columns for plotting individual trials
        TrialByTrialCols = (self.constants['ACCELERATION_X_COLUMN'], self.constants['ACCELERATION_Y_COLUMN'], 
                            self.constants['ACCELERATION_COLUMN'], self.constants['STIMULUS_COLUMN'],
                            self.constants['CORRECT_RESPONSE_COLUMN'], self.constants['IS_PRACTICE_COLUMN'], 
                            self.constants['TIME_COLUMN'])
        # Some of the necessay columns for plotting participant averages. 
        # Not all are shown since names are dynamic
        AveragedCols = ('PID', 'time')

        # Determine which DataFrame was inputted, and if necessary columns are present
        if all (key in DF.columns for key in TrialByTrialCols):
            Stimulus = DF[self.constants['STIMULUS_COLUMN']][participant]
            Correct_Response = DF[self.constants['CORRECT_RESPONSE_COLUMN']][participant]
            IsPractice = DF[self.constants['IS_PRACTICE_COLUMN']][participant]

            if ParentFig is None:
                plt.figure()
                plt.title('Stimulus: {}, Correct Response: {}, Is Practice Trial: {}'.format(Stimulus, Correct_Response, IsPractice))
            else:
                ParentFig

            for ax in axis:
                plt.plot(DF[self.constants['TIME_COLUMN']][participant], DF[self.constants[AxisMap[ax][0]]][participant], label = ax)

            if not HideLegend:
                plt.legend()
            plt.grid()
            if ShowAxis[0]:
                plt.xlabel('Time [ms]')
            else:
                plt.gca().set_xticklabels([])
            if ShowAxis[1]:
                plt.ylabel(r'Acceleration $\frac{m}{s^2}$')
            else:
                plt.gca().set_yticklabels([])
            if YLims is not None:
                plt.ylim(YLims)
            if XLims is not None:
                plt.xlim(XLims)
        elif all (key in DF.columns for key in AveragedCols):
            try:
                if not stimulus: 
                    stimulus = [DF.columns[-1].split(' ')[-1]]
                
                if ParentFig is None:
                    plt.figure()
                    plt.title('Participant: {}'.format(DF.loc[participant, 'PID']))
                else:
                    ParentFig

                for mov in movement:
                    for stim in stimulus:
                        col = 'Acc {} {}'.format(mov, stim)

                        for ax in axis:
                            t = DF.loc[participant, 'time']
                            y = DF.loc[participant, col][AxisMap[ax][1]]
                            reqShape = t.shape
                            plt.plot(t.T, y.reshape(reqShape).T, label = '{} {} ({})'.format(mov, stim, ax))
                            # plt.plot(t.T, y.reshape(reqShape).T, label = '{}'.format(stim))

                if not HideLegend:
                    plt.legend()
                plt.grid()
                if ShowAxis[0]:
                    plt.xlabel('Time [ms]')
                else:
                    plt.gca().set_xticklabels([])
                if ShowAxis[1]:
                    plt.ylabel(r'Acceleration $\frac{m}{s^2}$')
                else:
                    plt.gca().set_yticklabels([])
                if YLims is not None:
                    plt.ylim(YLims)
                if XLims is not None:
                    plt.xlim(XLims)
            except KeyError:
                print('[ERROR] - Inputted DataFrame contains columns "time" and "PID", so I assume it is a DF containing the averages of participants.')
                print('          However, no columns exist with the inputted movements ({}) or stimuli ({}).'.format(movement, stim))
        else:
            print('[ERROR] - Expected columns not present in the inputted DataFrame.')

        return None


    # Function to plot the displacement, in the x, y, and z directions, as a function of time
    # Units: Displacement in cm and Time in ms (milliseconds)
    def DistanceTime(self, participant, DF, Threshold = 60, axis = ['X', 'Y', 'Z'], movement = ['Pull', 'Push'], stimulus = None, ParentFig = None, ShowAxis = [True, True], YLims = None, XLims = None, HideLegend = False, **kwargs):
        AxisMap = {'X':['DISTANCE_X_COLUMN', 0], 
                   'Y':['DISTANCE_Y_COLUMN', 1], 
                   'Z':['DISTANCE_Z_COLUMN', 2]}

        # Necessary columns for plotting individual trials
        TrialByTrialCols = (self.constants['DISTANCE_X_COLUMN'], self.constants['DISTANCE_Y_COLUMN'], 
                            self.constants['DISTANCE_Z_COLUMN'], self.constants['STIMULUS_COLUMN'],
                            self.constants['CORRECT_RESPONSE_COLUMN'], self.constants['IS_PRACTICE_COLUMN'], 
                            self.constants['TIME_COLUMN'])
        # Some of the necessay columns for plotting participant averages. 
        # Not all are shown since names are dynamic
        AveragedCols = ('PID', 'time')

        # Determine which DataFrame was inputted, and if necessary columns are present
        if all (key in DF.columns for key in TrialByTrialCols):        
            Stimulus = DF[self.constants['STIMULUS_COLUMN']][participant]
            Correct_Response = DF[self.constants['CORRECT_RESPONSE_COLUMN']][participant]
            IsPractice = DF[self.constants['IS_PRACTICE_COLUMN']][participant]

            # Define data points which represent realistic limits of maximum distance
            Distance_UpperBound = DF[self.constants['TIME_COLUMN']][participant] * 0 + Threshold
            Distance_LowerBound = DF[self.constants['TIME_COLUMN']][participant] * 0 - Threshold

            max_dist = np.max([np.max(np.abs(DF[self.constants[AxisMap['X'][0]]][participant]*100)), np.max(np.abs(DF[self.constants[AxisMap['Y'][0]]][participant]*100)), np.max(np.abs(DF[self.constants[AxisMap['Z'][0]]][participant]*100))])

            if ParentFig is None:
                plt.figure()
                plt.title('Stimulus: {}, Correct Response: {}, Is Practice Trial: {}'.format(Stimulus, Correct_Response, IsPractice))
            else:
                ParentFig

            for ax in axis:
                plt.plot(DF[self.constants['TIME_COLUMN']][participant], DF[self.constants[AxisMap[ax][0]]][participant]*100, label = ax)
            
            # Plot threshold limits only if maximum distance is close to these limits
            if max_dist >= 0.8*Threshold:
                plt.plot(DF[self.constants['TIME_COLUMN']][participant], Distance_UpperBound, color = 'k', linestyle = '--', label = 'Maximum Realistic Distance')
                plt.plot(DF[self.constants['TIME_COLUMN']][participant], Distance_LowerBound, color = 'k', linestyle = '--')

            if not HideLegend:
                plt.legend()
            plt.grid()
            if ShowAxis[0]:
                plt.xlabel('Time [ms]')
            else:
                plt.gca().set_xticklabels([])
            if ShowAxis[1]:
                plt.ylabel(r'Distance [cm]')
            else:
                plt.gca().set_yticklabels([])
            if YLims is not None:
                plt.ylim(YLims)
            if XLims is not None:
                plt.xlim(XLims)

        elif all (key in DF.columns for key in AveragedCols):
            try:
                if not stimulus: 
                    stimulus = [DF.columns[-1].split(' ')[-1]]

                Distance_UpperBound = DF.loc[participant, 'time'] * 0 + Threshold
                Distance_LowerBound = DF.loc[participant, 'time'] * 0 - Threshold
                
                if ParentFig is None:
                    plt.figure()
                    plt.title('Participant: {}'.format(DF.loc[participant, 'PID']))
                else:
                    ParentFig

                maxDist = 0
                for mov in movement:
                    for stim in stimulus:
                        col = 'Dist {} {}'.format(mov, stim)

                        maxDist_i = 0
                        for ax in axis:
                            t = DF.loc[participant, 'time']
                            y = DF.loc[participant, col][AxisMap[ax][1]]*100
                            reqShape = t.shape
                            plt.plot(t.T, y.reshape(reqShape).T, label = '{} ({} {})'.format(ax, mov, stim))
                            maxDist_i = np.max([np.max(np.abs(y)), maxDist_i])
                        
                        maxDist = np.max([maxDist_i, maxDist])

                if maxDist >= 0.8*Threshold:
                    plt.plot(DF.loc[participant, 'time'], Distance_UpperBound, color = 'k', linestyle='--', label = 'Maximum Realistic Distance')
                    plt.plot(DF.loc[participant, 'time'], Distance_LowerBound, color = 'k', linestyle='--')

                if not HideLegend:
                    plt.legend()
                plt.grid()
                if ShowAxis[0]:
                    plt.xlabel('Time [ms]')
                else:
                    plt.gca().set_xticklabels([])
                if ShowAxis[1]:
                    plt.ylabel(r'Distance [cm]')
                else:
                    plt.gca().set_yticklabels([])
                if YLims is not None:
                    plt.ylim(YLims)
                if XLims is not None:
                    plt.xlim(XLims)

            except KeyError:
                print('[ERROR] - Inputted DataFrame contains columns "time" and "PID", so I assume it is a DF containing the averages of participants.')
                print('          However, no columns exist with the inputted movements ({}) or stimuli ({}).'.format(movement, stim))
        else:
            print('[ERROR] - Expected columns not present in the inputted DataFrame.')

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
    def Trajectory3D(self, participant, DF, Gradient = False, ColorMap = 'RdYlGn_r', movement = ['Pull', 'Push'], stimulus = None, ParentFig = None, **kwargs):
        AxisMap = {'X':['DISTANCE_X_COLUMN', 0], 
                   'Y':['DISTANCE_Y_COLUMN', 1], 
                   'Z':['DISTANCE_Z_COLUMN', 2]}        
        
        # Necessary columns for plotting individual trials
        TrialByTrialCols = (self.constants['DISTANCE_X_COLUMN'], self.constants['DISTANCE_Y_COLUMN'], 
                            self.constants['DISTANCE_Z_COLUMN'], self.constants['STIMULUS_COLUMN'],
                            self.constants['CORRECT_RESPONSE_COLUMN'], self.constants['TIME_COLUMN'])
        # Some of the necessay columns for plotting participant averages. 
        # Not all are shown since names are dynamic
        AveragedCols = ('PID', 'time')

        # Determine which DataFrame was inputted, and if necessary columns are present
        if all (key in DF.columns for key in TrialByTrialCols):      
            Stimulus = DF[self.constants['STIMULUS_COLUMN']][participant]
            Correct_Response = DF[self.constants['CORRECT_RESPONSE_COLUMN']][participant]

            X = DF[self.constants[AxisMap['X'][0]]][participant]*100
            Y = DF[self.constants[AxisMap['Y'][0]]][participant]*100
            Z = DF[self.constants[AxisMap['Z'][0]]][participant]*100
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

            if ParentFig is None:
                plt.figure()
                ax = plt.axes(projection = '3d')
                plt.title('Stimulus: {}, Correct Response: {}'.format(Stimulus, Correct_Response))
            else:
                ax = ParentFig
            
            # If we want to represent time as a color gradient on the trajectory
            if Gradient:
                cmap = mpl.cm.get_cmap(ColorMap)
                ax.add_collection3d(lc)
                # Indicate where the motion began and where it ended, since temporal information is not displayed 
                ax.scatter(Z[0], Y[0], X[0], c = [cmap(0)], label = 'Start of movement')
                ax.scatter(Z[-1], Y[-1], X[-1], c = [cmap(time[-1])], label = 'End of movement')
            else:
                ax.plot3D(Z, Y, X, label = 'Trajectory (displacement)')
                # Indicate where the motion began and where it ended, since temporal information is not displayed 
                ax.scatter(Z[0], Y[0], X[0], c = 'g', label = 'Start of movement')
                ax.scatter(Z[-1], Y[-1], X[-1], c = 'r', label = 'End of movement')
            
            ax.legend()
            ax.set_xlim3d(min_val, max_val)
            ax.set_ylim3d(min_val, max_val)
            ax.set_zlim3d(min_val, max_val)
            ax.set_xlabel('Apporach/Avoidance (z-axis) [cm]')
            ax.set_ylabel('Lateral Movement (y-axis) [cm]')
            ax.set_zlabel('Vertical Movement (x-axis) [cm]')

        elif all (key in DF.columns for key in AveragedCols):
            
            try:
                if not stimulus: 
                    stimulus = [DF.columns[-1].split(' ')[-1]]

                if ParentFig is None:
                    plt.figure()
                    ax = plt.axes(projection = '3d')
                    plt.title('Trajectory (displacement) for Participant: {}'.format(DF.loc[participant, 'PID']))
                else:
                    ax = ParentFig

                max_val = 0
                min_val = 0
                for mov in movement:
                    for stim in stimulus:
                        col = 'Dist {} {}'.format(mov, stim)
                        X = DF.loc[participant, col][AxisMap['X'][1]]*100
                        Y = DF.loc[participant, col][AxisMap['Y'][1]]*100
                        Z = DF.loc[participant, col][AxisMap['Z'][1]]*100
                        time = DF.loc[participant, 'time'].reshape((len(X),))

                        max_val = np.max([max_val, np.max([np.max(X), np.max(Y), np.max(Z)])])
                        min_val = np.min([min_val, np.min([np.min(X), np.min(Y), np.min(Z)])])

                        ax.plot3D(Z, Y, X, label = '{} {}'.format(mov, stim))
                        # Indicate where the motion began and where it ended, since temporal information is not displayed 
                        ax.scatter(Z[0], Y[0], X[0], c = 'g')
                        ax.scatter(Z[-1], Y[-1], X[-1], c = 'r')
                
                ax.scatter(Z[0], Y[0], X[0], c = 'g', label = 'Start of movement')
                ax.scatter(Z[-1], Y[-1], X[-1], c = 'r', label = 'End of movement')
                        
                ax.legend()
                ax.set_xlim3d(min_val, max_val)
                ax.set_ylim3d(min_val, max_val)
                ax.set_zlim3d(min_val, max_val)
                ax.set_xlabel('Apporach/Avoidance (z-axis) [cm]')
                ax.set_ylabel('Lateral Movement (y-axis) [cm]')
                ax.set_zlabel('Vertical Movement (x-axis) [cm]')

            except KeyError:
                print('[ERROR] - Inputted DataFrame contains columns "time" and "PID", so I assume it is a DF containing the averages of participants.')
                print('          However, no columns exist with the inputted movements ({}) or stimuli ({}).'.format(movement, stim))
        else:
            print('[ERROR] - Expected columns not present in the inputted DataFrame.')

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
    def Acceleration3D(self, participant, DF, Gradient = False, ColorMap = 'RdYlGn_r', movement = ['Pull', 'Push'], stimulus = None, ParentFig = None, **kwargs):
        AxisMap = {'X':['ACCELERATION_X_COLUMN', 0], 
                   'Y':['ACCELERATION_Y_COLUMN', 1], 
                   'Z':['ACCELERATION_COLUMN', 2]} 

        # Necessary columns for plotting individual trials
        TrialByTrialCols = (self.constants['ACCELERATION_X_COLUMN'], self.constants['ACCELERATION_Y_COLUMN'], 
                            self.constants['ACCELERATION_COLUMN'], self.constants['STIMULUS_COLUMN'],
                            self.constants['CORRECT_RESPONSE_COLUMN'], self.constants['TIME_COLUMN'])
        # Some of the necessay columns for plotting participant averages. 
        # Not all are shown since names are dynamic
        AveragedCols = ('PID', 'time')

        # Determine which DataFrame was inputted, and if necessary columns are present
        if all (key in DF.columns for key in TrialByTrialCols):
            Stimulus = DF[self.constants['STIMULUS_COLUMN']][participant]
            Correct_Response = DF[self.constants['CORRECT_RESPONSE_COLUMN']][participant]

            X = DF[self.constants[AxisMap['X'][0]]][participant]
            Y = DF[self.constants[AxisMap['Y'][0]]][participant]
            Z = DF[self.constants[AxisMap['Z'][0]]][participant]
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

            if ParentFig is None:
                plt.figure()
                ax = plt.axes(projection = '3d')
                plt.title('Stimulus: {}, Correct Response: {}'.format(Stimulus, Correct_Response))
            else:
                ax = ParentFig        

            # If we want to represent time as a color gradient on the trajectory
            if Gradient:
                cmap = mpl.cm.get_cmap(ColorMap)
                ax.add_collection3d(lc)
                # Indicate where the motion began and where it ended, since temporal information is not displayed 
                ax.scatter(Z[0], Y[0], X[0], c = [cmap(0)], label = 'Start of movement')
                ax.scatter(Z[-1], Y[-1], X[-1], c = [cmap(time[-1])], label = 'End of movement')
            else:
                ax.plot3D(Z, Y, X, label = 'Acceleration')
                # Indicate where the motion began and where it ended, since temporal information is not displayed 
                ax.scatter(Z[0], Y[0], X[0], c = 'g', label = 'Start of movement')
                ax.scatter(Z[-1], Y[-1], X[-1], c = 'r', label = 'End of movement')

            ax.legend()
            ax.set_xlim3d(min_val, max_val)
            ax.set_ylim3d(min_val, max_val)
            ax.set_zlim3d(min_val, max_val)
            ax.set_xlabel(r'Apporach/Avoidance (z-axis) $[\frac{m}{s^2}]$')
            ax.set_ylabel(r'Lateral Movement (y-axis) $[\frac{m}{s^2}]$')
            ax.set_zlabel(r'Vertical Movement (x-axis) $[\frac{m}{s^2}]$')

        elif all (key in DF.columns for key in AveragedCols):
    
            try:
                if not stimulus: 
                    stimulus = [DF.columns[-1].split(' ')[-1]]

                if ParentFig is None:
                    plt.figure()
                    ax = plt.axes(projection = '3d')
                    plt.title('3-D Acceleration for Participant: {}'.format(DF.loc[participant, 'PID']))
                else:
                    ax = ParentFig

                max_val = 0
                min_val = 0
                for mov in movement:
                    for stim in stimulus:
                        col = 'Acc {} {}'.format(mov, stim)
                        X = DF.loc[participant, col][AxisMap['X'][1]]
                        Y = DF.loc[participant, col][AxisMap['Y'][1]]
                        Z = DF.loc[participant, col][AxisMap['Z'][1]]
                        time = DF.loc[participant, 'time'].reshape((len(X),))

                        max_val = np.max([max_val, np.max([np.max(X), np.max(Y), np.max(Z)])])
                        min_val = np.min([min_val, np.min([np.min(X), np.min(Y), np.min(Z)])])

                        ax.plot3D(Z, Y, X, label = '{} {}'.format(mov, stim))
                        # Indicate where the motion began and where it ended, since temporal information is not displayed 
                        ax.scatter(Z[0], Y[0], X[0], c = 'g')
                        ax.scatter(Z[-1], Y[-1], X[-1], c = 'r')
                
                ax.scatter(Z[0], Y[0], X[0], c = 'g', label = 'Start of movement')
                ax.scatter(Z[-1], Y[-1], X[-1], c = 'r', label = 'End of movement')

                ax.legend()
                ax.set_xlim3d(min_val, max_val)
                ax.set_ylim3d(min_val, max_val)
                ax.set_zlim3d(min_val, max_val)
                ax.set_xlabel(r'Apporach/Avoidance (z-axis) $[\frac{m}{s^2}]$')
                ax.set_ylabel(r'Lateral Movement (y-axis) $[\frac{m}{s^2}]$')
                ax.set_zlabel(r'Vertical Movement (x-axis) $[\frac{m}{s^2}]$')

            except KeyError:
                print('[ERROR] - Inputted DataFrame contains columns "time" and "PID", so I assume it is a DF containing the averages of participants.')
                print('          However, no columns exist with the inputted movements ({}) or stimuli ({}).'.format(movement, stim))
        else:
            print('[ERROR] - Expected columns not present in the inputted DataFrame.')
        

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
    def AnimateTrajectory3D(self, participant, DF, UseBlit=True, View=[20, -60], Save=False, movement = ['Pull', 'Push'], stimulus = None):
        # Function to update data being shown in the plot (single line)
        def update(num, data, line):
            line.set_data(data[:2, :num])
            line.set_3d_properties(data[2, :num])
            return line, 
        
        # Function to update data being shown in the plot (several lines)
        def updateMulti(num, data, line):
            for l, d in zip(line, data):
                l.set_data(d[:2, :num])
                l.set_3d_properties(d[2, :num])
            return line

        AxisMap = {'X':['DISTANCE_X_COLUMN', 0], 
            'Y':['DISTANCE_Y_COLUMN', 1], 
            'Z':['DISTANCE_Z_COLUMN', 2]}   

        # Necessary columns for plotting individual trials
        TrialByTrialCols = (self.constants['DISTANCE_X_COLUMN'], self.constants['DISTANCE_Y_COLUMN'], 
                            self.constants['DISTANCE_Z_COLUMN'], self.constants['STIMULUS_COLUMN'],
                            self.constants['CORRECT_RESPONSE_COLUMN'], self.constants['TIME_COLUMN'])
        # Some of the necessay columns for plotting participant averages. 
        # Not all are shown since names are dynamic
        AveragedCols = ('PID', 'time')

        # Determine which DataFrame was inputted, and if necessary columns are present
        if all (key in DF.columns for key in TrialByTrialCols):    
            Stimulus = DF[self.constants['STIMULUS_COLUMN']][participant]
            Correct_Response = DF[self.constants['CORRECT_RESPONSE_COLUMN']][participant]

            X = DF[self.constants[AxisMap['X'][0]]][participant]*100
            Y = DF[self.constants[AxisMap['Y'][0]]][participant]*100
            Z = DF[self.constants[AxisMap['Z'][0]]][participant]*100
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
        
        elif all (key in DF.columns for key in AveragedCols):

            try:
                if not stimulus: 
                    stimulus = [DF.columns[-1].split(' ')[-1]]

                dataLines = []
                labels = []

                max_val = 0
                min_val = 0
                for mov in movement:
                    for stim in stimulus:
                        col = 'Dist {} {}'.format(mov, stim)
                        X = DF.loc[participant, col][AxisMap['X'][1]]
                        Y = DF.loc[participant, col][AxisMap['Y'][1]]
                        Z = DF.loc[participant, col][AxisMap['Z'][1]]
                        time = DF.loc[participant, 'time'].reshape((len(X),))

                        max_val = np.max([max_val, np.max([np.max(X), np.max(Y), np.max(Z)])])
                        min_val = np.min([min_val, np.min([np.min(X), np.min(Y), np.min(Z)])])

                        dataLines.append([Z, Y, X])
                        labels.append('{} {}'.format(mov, stim))

                MovFig = plt.figure()
                ax = mplot3d.axes3d.Axes3D(MovFig)

                dataLines = np.array(dataLines)
                lines = [ax.plot(data[0, 0:1], data[1, 0:1], data[2, 0:1], label=labels[idx])[0] for idx, data in enumerate(dataLines)]

                ax.set_xlabel('Apporach/Avoidance (z-axis) [cm]')
                ax.set_ylabel('Lateral Movement (y-axis) [cm]')
                ax.set_zlabel('Vertical Movement (x-axis) [cm]')
                ax.set_xlim3d(min_val, max_val)
                ax.set_ylim3d(min_val, max_val)
                ax.set_zlim3d(min_val, max_val)

                # Set viewpoint; elev = Elevation in degrees; azim = Azimuth in degrees
                ax.view_init(elev=View[0], azim=View[1])
                
                ax.legend(loc='lower left', bbox_to_anchor = (0.7, 0.7))

                line_animation = animation.FuncAnimation(MovFig, updateMulti, len(time), fargs=(dataLines, lines), interval=1, blit=UseBlit)
                plt.show()

                if Save:
                    line_animation.save('Trajectory_{}_{}_{}.mp4'.format(DF.loc[participant, 'PID'], mov, stim), writer='FFMpegWriter') 
            
            except KeyError:
                print('[ERROR] - Inputted DataFrame contains columns "time" and "PID", so I assume it is a DF containing the averages of participants.')
                print('          However, no columns exist with the inputted movements ({}) or stimuli ({}).'.format(movement, stim))
        else:
            print('[ERROR] - Expected columns not present in the inputted DataFrame.')
        
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
    def AnimateAcceleration3D(self, participant, DF, UseBlit=True, View=[20, -60], Save=False, movement = ['Pull', 'Push'], stimulus = None):
        # Function to update data being shown in the plot
        def update(num, data, line):
            line.set_data(data[:2, :num])
            line.set_3d_properties(data[2, :num])
            return line, 

        # Function to update data being shown in the plot (several lines)
        def updateMulti(num, data, line):
            for l, d in zip(line, data):
                l.set_data(d[:2, :num])
                l.set_3d_properties(d[2, :num])
            return line

        AxisMap = {'X':['ACCELERATION_X_COLUMN', 0], 
                   'Y':['ACCELERATION_Y_COLUMN', 1], 
                   'Z':['ACCELERATION_COLUMN', 2]}

        # Necessary columns for plotting individual trials
        TrialByTrialCols = (self.constants['ACCELERATION_X_COLUMN'], self.constants['ACCELERATION_Y_COLUMN'], 
                            self.constants['ACCELERATION_COLUMN'], self.constants['STIMULUS_COLUMN'],
                            self.constants['CORRECT_RESPONSE_COLUMN'], self.constants['TIME_COLUMN'])
        # Some of the necessay columns for plotting participant averages. 
        # Not all are shown since names are dynamic
        AveragedCols = ('PID', 'time')

        # Determine which DataFrame was inputted, and if necessary columns are present
        if all (key in DF.columns for key in TrialByTrialCols):
            Stimulus = DF[self.constants['STIMULUS_COLUMN']][participant]
            Correct_Response = DF[self.constants['CORRECT_RESPONSE_COLUMN']][participant]

            X = DF[self.constants[AxisMap['X'][0]]][participant]
            Y = DF[self.constants[AxisMap['Y'][0]]][participant]
            Z = DF[self.constants[AxisMap['Z'][0]]][participant]
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

        elif all (key in DF.columns for key in AveragedCols):
    
            try:
                if not stimulus: 
                    stimulus = [DF.columns[-1].split(' ')[-1]]

                dataLines = []
                labels = []

                max_val = 0
                min_val = 0
                for mov in movement:
                    for stim in stimulus:
                        col = 'Acc {} {}'.format(mov, stim)
                        X = DF.loc[participant, col][AxisMap['X'][1]]
                        Y = DF.loc[participant, col][AxisMap['Y'][1]]
                        Z = DF.loc[participant, col][AxisMap['Z'][1]]
                        time = DF.loc[participant, 'time'].reshape((len(X),))

                        max_val = np.max([max_val, np.max([np.max(X), np.max(Y), np.max(Z)])])
                        min_val = np.min([min_val, np.min([np.min(X), np.min(Y), np.min(Z)])])

                        dataLines.append([Z, Y, X])
                        labels.append('{} {}'.format(mov, stim))

                MovFig = plt.figure()
                ax = mplot3d.axes3d.Axes3D(MovFig)

                dataLines = np.array(dataLines)
                lines = [ax.plot(data[0, 0:1], data[1, 0:1], data[2, 0:1], label=labels[idx])[0] for idx, data in enumerate(dataLines)]

                ax.set_xlabel(r'Apporach/Avoidance (z-axis) $[\frac{m}{s^2}]$')
                ax.set_ylabel(r'Lateral Movement (y-axis) $[\frac{m}{s^2}]$')
                ax.set_zlabel(r'Vertical Movement (x-axis) $[\frac{m}{s^2}]$')
                ax.set_xlim3d(min_val, max_val)
                ax.set_ylim3d(min_val, max_val)
                ax.set_zlim3d(min_val, max_val)

                # Set viewpoint; elev = Elevation in degrees; azim = Azimuth in degrees
                ax.view_init(elev=View[0], azim=View[1])
                
                ax.legend(loc='lower left', bbox_to_anchor = (0.7, 0.7))

                line_animation = animation.FuncAnimation(MovFig, updateMulti, len(time), fargs=(dataLines, lines), interval=1, blit=UseBlit)
                plt.show()

                if Save:
                    line_animation.save('3DAcceleration_{}_{}_{}.mp4'.format(DF.loc[participant, 'PID'], mov, stim), writer='FFMpegWriter') 
            
            except KeyError:
                print('[ERROR] - Inputted DataFrame contains columns "time" and "PID", so I assume it is a DF containing the averages of participants.')
                print('          However, no columns exist with the inputted movements ({}) or stimuli ({}).'.format(movement, stim))
        else:
            print('[ERROR] - Expected columns not present in the inputted DataFrame.')
        
        return None


    def ApproachAvoidanceXZ(self, metric, participant, DF, movement = ['Pull', 'Push'], stimulus = None, ParentFig = None, **kwargs):
        if metric.lower() == 'acceleration':
            AxisMap = {'X':['ACCELERATION_X_COLUMN', 0],
                    'Z':['ACCELERATION_COLUMN', 2]}
            Lab = r'Acceleration $\frac{m}{s^2}$'
            Factor = 1
            met = 'Acc'
            showRegion = False
        elif metric.lower() == 'distance':
            AxisMap = {'X':['DISTANCE_X_COLUMN', 0],
                   'Z':['DISTANCE_Z_COLUMN', 2]} 
            Lab = 'Distance [cm]'
            Factor = 100
            met = 'Dist'
            showRegion = True
        else:
            print('[ERROR] <metric> not understood in <ApproachAvoidanceXZ>. User inputted <{}>, Please use <acceleration> or <distance>.'.format(metric))
            AxisMap = None

        if AxisMap is not None:
            # Necessary columns for plotting individual trials
            TrialByTrialCols = (self.constants[AxisMap['X'][0]], self.constants[AxisMap['Z'][0]], 
                                self.constants['STIMULUS_COLUMN'], self.constants['CORRECT_RESPONSE_COLUMN'], 
                                self.constants['TIME_COLUMN'], self.constants['IS_PRACTICE_COLUMN'])
            # Some of the necessay columns for plotting participant averages. 
            # Not all are shown since names are dynamic
            AveragedCols = ('PID', 'time')

            # Determine which DataFrame was inputted, and if necessary columns are present
            if all (key in DF.columns for key in TrialByTrialCols):
                Stimulus = DF[self.constants['STIMULUS_COLUMN']][participant]
                Correct_Response = DF[self.constants['CORRECT_RESPONSE_COLUMN']][participant]
                IsPractice = DF[self.constants['IS_PRACTICE_COLUMN']][participant]

                if ParentFig is None:
                    plt.figure()
                    plt.title('Stimulus: {}, Correct Response: {}, Is Practice Trial: {}'.format(Stimulus, Correct_Response, IsPractice))
                else:
                    ParentFig

                ax = plt.gca()

                if showRegion:
                    theta = np.linspace(0, 2*np.pi, 100)
                    r = 20
                    x1 = r * np.cos(theta) + r
                    x2 = r * np.sin(theta)

                    avoidColor = (0.8, 0.0, 0.0, 0.2)
                    approachColor = (0.0, 1.0, 0., 0.4)

                    ax.set_facecolor(avoidColor)
                    plt.fill_between(x1, x2, color = approachColor[:-1], alpha = 0.15)
                    plt.fill_between(x1, x2, color = approachColor[:-1], alpha = 0.15)

                startColor = 'purple'
                endColor = 'yellow'

                z = DF[self.constants[AxisMap['Z'][0]]][participant]*Factor
                x = DF[self.constants[AxisMap['X'][0]]][participant]*Factor

                minx = np.min(x)
                maxx = np.max(x)
                minz = np.min(z)
                maxz = np.max(z)

                plt.plot(z, x, label = 'X-Z Movement')
                plt.scatter(z[0], x[0], color = startColor, label = 'Start of movement')
                plt.scatter(z[-1], x[-1], color = endColor, label = 'End of movement')

                handles, labels = ax.get_legend_handles_labels()
                if showRegion:
                    approachPatch = mpl.patches.Patch(color = approachColor, label = '(Approximate) Approach region')
                    avoidancePatch = mpl.patches.Patch(color = avoidColor, label = '(Approximate) Avoid region')
                    handles.append(approachPatch)
                    handles.append(avoidancePatch)

                plt.legend(handles = handles)
                plt.grid()
                plt.xlabel(r'Z (Ventral Axis) {}'.format(Lab))
                plt.ylabel(r'X (Vertical Axis) {}'.format(Lab))
                plt.xlim([1.1*minz, 1.1*maxz])
                plt.ylim([1.1*minx, 1.1*maxx])

            elif all (key in DF.columns for key in AveragedCols):
        
                try:
                    if not stimulus: 
                        stimulus = [DF.columns[-1].split(' ')[-1]]
                    
                    if ParentFig is None:
                        plt.figure()
                        plt.title('Participant: {}'.format(DF.loc[participant, 'PID']))
                    else:
                        ParentFig

                    ax = plt.gca()

                    if showRegion:
                        theta = np.linspace(0, 2*np.pi, 100)
                        r = 20
                        x1 = r * np.cos(theta) + r
                        x2 = r * np.sin(theta)

                        avoidColor = (0.8, 0.0, 0.0, 0.2)
                        approachColor = (0.0, 1.0, 0., 0.4)

                        ax.set_facecolor(avoidColor)
                        plt.fill_between(x1, x2, color = approachColor[:-1], alpha = 0.15)
                        plt.fill_between(x1, x2, color = approachColor[:-1], alpha = 0.15)

                    minx = 0
                    maxx = 0
                    minz = 0
                    maxz = 0

                    startColor = 'purple'
                    endColor = 'yellow'

                    for mov in movement:
                        for stim in stimulus:
                            col = '{} {} {}'.format(met, mov, stim)
                            x = DF.loc[participant, col][AxisMap['X'][1]] * Factor
                            z = DF.loc[participant, col][AxisMap['Z'][1]] * Factor
                            plt.plot(z, x, label = '({} {})'.format(mov, stim))
                            plt.scatter(z[0], x[0], color = startColor)
                            plt.scatter(z[-1], x[-1], color = endColor)
                            minx = np.min([minx, np.min(x)])
                            maxx = np.max([maxx, np.max(x)])
                            minz = np.min([minz, np.min(z)])
                            maxz = np.max([maxz, np.max(z)])

                    plt.scatter(z[0], x[0], color = startColor, label = 'Start of movement')
                    plt.scatter(z[-1], x[-1], color = endColor, label = 'End of movement')
                    
                    handles, labels = ax.get_legend_handles_labels()
                    if showRegion:
                        approachPatch = mpl.patches.Patch(color = approachColor, label = '(Approximate) Approach region')
                        avoidancePatch = mpl.patches.Patch(color = avoidColor, label = '(Approximate) Avoid region')
                        handles.append(approachPatch)
                        handles.append(avoidancePatch)

                    plt.legend(handles = handles)
                    plt.grid()
                    plt.xlabel(r'Z (Ventral Axis) {}'.format(Lab))
                    plt.ylabel(r'X (Vertical Axis) {}'.format(Lab))
                    plt.xlim([1.1*minz, 1.1*maxz])
                    plt.ylim([1.1*minx, 1.1*maxx])
                
                except KeyError:
                    print('[ERROR] - Inputted DataFrame contains columns "time" and "PID", so I assume it is a DF containing the averages of participants.')
                    print('          However, no columns exist with the inputted movements ({}) or stimuli ({}).'.format(movement, stim))
            else:
                print('[ERROR] - Expected columns not present in the inputted DataFrame.')
        
        return None


    def MultiPlot(self, Layout, Functions, FuncArgs):

        # Utility function to get the value of an input argument
        # For optional arguments, in case a value is not given
        # by the user, the function will attempt to find the 
        # default value and return that. 
        def getParam(InputtedParams, key, f=None):
            try:
                param = InputtedParams[key]
            except KeyError:
                # For the case that we want to keep the default kwarg
                if f is not None:
                    num_input_args = f.__code__.co_argcount
                    # Start at 1 since input argument at index 0 is self
                    f_all_inputs = f.__code__.co_varnames[1:num_input_args]
                    f_defaults = f.__defaults__
                    f_args = f_all_inputs[-len(f_defaults):]
                    input_params = {k:f_defaults[v] for v,k in enumerate(f_args)}
                    param = getParam(input_params, key)
                else:
                    param = None
            return param

        fig = plt.figure()

        for idx, func in enumerate(Functions):
            # Make sure that a function is present
            if func is not None:
                # Need to create special subplot for 3D graphs. All 3D functions have '3D' at the end
                # so we use this to identify them 
                if func.__name__.endswith('3D'):
                    subFig = fig.add_subplot(Layout[0], Layout[1], idx + 1, projection = '3d')
                else:
                    subFig = plt.subplot(Layout[0], Layout[1], idx + 1)
                Params = FuncArgs[idx]
                func(participant = getParam(Params, 'participant'), DF = getParam(Params, 'DF'), ParentFig = subFig,
                    metric = getParam(Params, 'metric', func), axis = getParam(Params, 'axis', func), 
                    movement = getParam(Params, 'movement', func), stimulus = getParam(Params, 'stimulus', func),
                    Gradient = getParam(Params, 'Gradient', func), ColorMap = getParam(Params, 'ColorMap', func), 
                    Threshold = getParam(Params, 'Threshold', func), YLims = getParam(Params, 'YLims', func), 
                    ShowAxis = getParam(Params, 'ShowAxis', func), HideLegend = getParam(Params, 'HideLegend', func),
                    XLims = getParam(Params, 'XLims', func))

        return None



# Class to handle analysis of the preprocessed raw data
class Analysis:

    def __init__(self, Control, Target, constants = None, printINFO = True):
        # If no constants are given, take them from the DataImporter Class
        if constants is None:
            Importer = DataImporter('DummyPath', 'DummyPath')
            self.constants = Importer.constants
        else:
            self.constants = constants

        self.INFO = printINFO
        self.Control = Control
        self.Target = Target


    def FuseStimulusSets(self, DataFrame, Mapping, sgParams = None):
        DF = DataFrame.copy(deep = True)
        stimSetCol = self.constants['STIMULUS_SET_COLUMN']
        sessionCol = self.constants['SESSION_COLUMN']

        idxs = {'{}'.format(self.Target):[], '{}'.format(self.Control):[]}

        execution_time = 0
        sTime = time.time()

        N = 0
        for session in Mapping.keys():
            for stimset in Mapping[session].keys():
                N += 1
        
        i = 0
        for session in Mapping.keys():
            for stimset in Mapping[session].keys():
                if sgParams:
                    sgParams['PERCENTAGE'].Update('{:2d} %'.format(int(100*(i+1)/N)))
                    sgParams['PROG_BAR'].UpdateBar(i + 1, N)
                    timeRemainingS = execution_time/(i + 1) * (N - i + 1)
                    sgParams['TIME'].Update('Time remaining:{:6d} [s]'.format(int(timeRemainingS)))
                    events, values = sgParams['WINDOW'].read(timeout=1)
                    if events in (sg.WIN_CLOSED, '-CANCEL-'):
                        sgParams.update({'IS_CLOSED':True})
                        break
                try:
                    idxList = idxs[Mapping[session][stimset]]
                    sessionIdx = np.where(DF[sessionCol].str.match(session))[0]
                    stimIdx = np.where(DF[stimSetCol].str.match(stimset))[0]
                    idx = np.intersect1d(sessionIdx, stimIdx)
                    # *idxList unpacks idxList, and *idx unpacks idx, the result is a combined list of indexes
                    idxList = [*idxList, *idx]
                    idxs.update({Mapping[session][stimset]:idxList})
                except KeyError:
                    print('[ ERROR ] Please check Mapping. The provided mapping element: {} is not compatible with the target: {} or control: {}'.format(Mapping[session][stimset], self.Target, self.Control))
                    DF = None
                    break
                i += 1
                execution_time = time.time() - sTime
        
        if DF is not None:
            DF.at[idxs[self.Target], stimSetCol] = self.Target
            DF.at[idxs[self.Control], stimSetCol] = self.Control

        return DF


    # Function to compute the mean and standard deviation of the 
    # various independent variables (Push/Pull and Stimulus Type)
    # and their interactions (e.g. Push Control Stimulus) for
    # Reaction time, Peak acceleration and Peak distance
    # Input:    DataFrame = DataFrame containing the processed AAT data
    #           Save2CSV = Boolean to save the statistics to a csv file
    #           SavePath = Path to save location, if None, then the 
    #                      program will save file in the same directory
    #                      as the program.
    # Output:   Stats = List of dataframes containing the general 
    #                   statistics for the reaction time, peak 
    #                   acceleration and peak distance 
    def GeneralStats(self, DataFrame, Save2CSV = False, SavePath = None, sgParams = None):

        def ComputeStats(header, col, Control, Target, PullIdx, PushIdx, ControlIdx, TargetIdx):
            x = DataFrame[col]

            intBothIdx = list(np.where(ControlIdx)[0]) + list(np.where(TargetIdx)[0])
            bothIdx = np.array([False] * len(PushIdx))
            bothIdx[intBothIdx] = True

            variables = {'Push':(PushIdx & bothIdx), 'Pull':(PullIdx & bothIdx), '{}'.format(Control):ControlIdx, 
                         '{}'.format(Target):TargetIdx, 'Push x {}'.format(Control):(ControlIdx & PushIdx),
                         'Pull x {}'.format(Control):(ControlIdx & PullIdx), 'Push x {}'.format(Target):(TargetIdx + PushIdx), 
                         'Pull x {}'.format(Target):(TargetIdx & PullIdx)}

            headers = ['condition', 'mean', 'std']

            data = np.zeros((len(variables.keys()), len(headers) - 1))
            
            for idx, key in enumerate(variables.keys()):
                data[idx, 0] = np.nanmean(x[variables[key]])
                data[idx, 1] = np.nanstd(x[variables[key]])

            Tab = pd.DataFrame(data = data, columns=headers[1:], index = variables)
            
            print('\n')
            print('===============================================================')
            print('{} statistics'.format(header))
            print('===============================================================')
            print('Condition\t\t|\tMean\t\t|\tstd')
            print('Push\t\t\t|\t{:.2f}\t\t|\t{:.2f}'.format(np.nanmean(x[PushIdx]), np.nanstd(x[PushIdx])))
            print('Pull\t\t\t|\t{:.2f}\t\t|\t{:.2f}'.format(np.nanmean(x[PullIdx]), np.nanstd(x[PullIdx])))
            print('{}\t\t\t|\t{:.2f}\t\t|\t{:.2f}'.format(Control, np.nanmean(x[ControlIdx]), np.nanstd(x[ControlIdx])))
            print('{}\t\t\t|\t{:.2f}\t\t|\t{:.2f}'.format(Target, np.nanmean(x[TargetIdx]), np.nanstd(x[TargetIdx])))
            print('Push x {}\t\t|\t{:.2f}\t\t|\t{:.2f}'.format(Control, np.nanmean(x[ControlIdx & PushIdx]), np.nanstd(x[ControlIdx & PushIdx])))
            print('Pull x {}\t\t|\t{:.2f}\t\t|\t{:.2f}'.format(Control, np.nanmean(x[ControlIdx & PullIdx]), np.nanstd(x[ControlIdx & PullIdx])))
            print('Push x {}\t\t|\t{:.2f}\t\t|\t{:.2f}'.format(Target, np.nanmean(x[TargetIdx & PushIdx]), np.nanstd(x[TargetIdx & PushIdx])))
            print('Pull x {}\t\t|\t{:.2f}\t\t|\t{:.2f}'.format(Target, np.nanmean(x[TargetIdx & PullIdx]), np.nanstd(x[TargetIdx & PullIdx])))
            print('===============================================================')
            print('\n')

            if sgParams is not None:
                sgParams['OUTPUT'].print('\n')
                sgParams['OUTPUT'].print('===============================================================')
                sgParams['OUTPUT'].print('{} statistics'.format(header))
                sgParams['OUTPUT'].print('===============================================================')
                sgParams['OUTPUT'].print('Condition\t\t|\tMean\t\t|\tstd')
                sgParams['OUTPUT'].print('Push\t\t|\t{:.2f}\t\t|\t{:.2f}'.format(np.nanmean(x[PushIdx]), np.nanstd(x[PushIdx])))
                sgParams['OUTPUT'].print('Pull\t\t|\t{:.2f}\t\t|\t{:.2f}'.format(np.nanmean(x[PullIdx]), np.nanstd(x[PullIdx])))
                sgParams['OUTPUT'].print('{}\t\t|\t{:.2f}\t\t|\t{:.2f}'.format(Control, np.nanmean(x[ControlIdx]), np.nanstd(x[ControlIdx])))
                sgParams['OUTPUT'].print('{}\t\t|\t{:.2f}\t\t|\t{:.2f}'.format(Target, np.nanmean(x[TargetIdx]), np.nanstd(x[TargetIdx])))
                sgParams['OUTPUT'].print('Push x {}\t\t|\t{:.2f}\t\t|\t{:.2f}'.format(Control, np.nanmean(x[ControlIdx & PushIdx]), np.nanstd(x[ControlIdx & PushIdx])))
                sgParams['OUTPUT'].print('Pull x {}\t\t|\t{:.2f}\t\t|\t{:.2f}'.format(Control, np.nanmean(x[ControlIdx & PullIdx]), np.nanstd(x[ControlIdx & PullIdx])))
                sgParams['OUTPUT'].print('Push x {}\t\t|\t{:.2f}\t\t|\t{:.2f}'.format(Target, np.nanmean(x[TargetIdx & PushIdx]), np.nanstd(x[TargetIdx & PushIdx])))
                sgParams['OUTPUT'].print('Pull x {}\t\t|\t{:.2f}\t\t|\t{:.2f}'.format(Target, np.nanmean(x[TargetIdx & PullIdx]), np.nanstd(x[TargetIdx & PullIdx])))
                sgParams['OUTPUT'].print('===============================================================')
                sgParams['OUTPUT'].print('\n')

            return Tab

        # If no save path is provided, set it to the program folder
        if SavePath is None:
            SavePath = os.getcwd()

        # Inherit control and target from class initialization 
        Control = self.Control
        Target = self.Target

        PullIdx = (DataFrame[self.constants['CORRECT_RESPONSE_COLUMN']]=='Pull')
        PushIdx = (DataFrame[self.constants['CORRECT_RESPONSE_COLUMN']]=='Push')
        ControlIdx = (DataFrame[self.constants['STIMULUS_SET_COLUMN']]==Control)
        TargetIdx = (DataFrame[self.constants['STIMULUS_SET_COLUMN']]==Target)
        
        print('[INFO] GENERAL STATISTICS')
        if sgParams is not None:
            sgParams['OUTPUT'].print('[ INFO ] General statistics')

        Stats = {}

        RTStats = ComputeStats('Reaction time [ms]', self.constants['RT_COLUMN'], Control, Target, 
                              PullIdx.to_numpy(), PushIdx.to_numpy(), ControlIdx.to_numpy(), TargetIdx.to_numpy())
        Stats.update({'RT':RTStats})
        if self.constants['DELTA_A_COLUMN'] in DataFrame.columns:  
            DAStats = ComputeStats('Peak acceleration [m/s^-2]', self.constants['DELTA_A_COLUMN'], Control, Target, 
                                PullIdx.to_numpy(), PushIdx.to_numpy(), ControlIdx.to_numpy(), TargetIdx.to_numpy())
            Stats.update({'Acceleration':DAStats})
        if self.constants['DELTA_D_COLUMN'] in DataFrame.columns:  
            DDStats = ComputeStats('Peak displacement [m]', self.constants['DELTA_D_COLUMN'], Control, Target, 
                                PullIdx.to_numpy(), PushIdx.to_numpy(), ControlIdx.to_numpy(), TargetIdx.to_numpy())
            Stats.update({'Distance':DDStats})
        
        if Save2CSV:
            print('[INFO] Saving general statistics in directory: {}'.format(SavePath))
            

        return Stats



    def LinearMixedModels(self, DataFrame, Save2CSV = False, Verbose = False, SavePath = None, ReturnModel = False, ReturnModelSummary = False, sgParams = None):
        ExpectedCols = [self.constants['STIMULUS_SET_COLUMN'], self.constants['CORRECT_RESPONSE_COLUMN'],
                        self.constants['RT_COLUMN'], self.constants['PARTICIPANT_COLUMN']]

        Control = self.Control
        Target = self.Target

        if SavePath is None:
            SavePath = os.getcwd()

        if Save2CSV:
            print('[INFO] Save file directioy: {}'.format(SavePath))

        DF = DataFrame.copy(deep = True)

        models = {}

        if all (key in DataFrame.columns for key in ExpectedCols):
            # Remove practice trials:
            DF = DF[(~DF[self.constants['STIMULUS_SET_COLUMN']].str.contains('practice')) 
                       & (~DF[self.constants['STIMULUS_SET_COLUMN']].str.contains('practice'))]

            # Specify the independent variables
            DF['is_{}'.format(Target)] = DF[self.constants['STIMULUS_SET_COLUMN']]
            DF['is_{}'.format(Target)].replace({'{}'.format(Control):0, '{}'.format(Target):1}, inplace = True)

            # Remove all stimuli other than the target and control stimuli
            TarIdx = np.where(DF['is_{}'.format(Target)] == 1)
            ConIdx = np.where(DF['is_{}'.format(Target)] == 0)
            RelevantIdx = np.unique(np.hstack((TarIdx, ConIdx)))
            DF = DF.iloc[RelevantIdx, :]

            # Define pull idx
            DF['is_pull'] = DF[self.constants['CORRECT_RESPONSE_COLUMN']]
            DF['is_pull'].replace({'Push':0, 'Pull':1}, inplace = True)

            # Mirror the values around 0, such that congruent motions lead to positive effects
            # Consider, for example the following case: 
            #       Pull = 1, Push = 0 and Happy = 1, Angry = 0
            # We expect that congruent motions (e.g pull happy and push angry) lead to faster reactions
            # Let us tabulate the product of the combinations of the above values
            #       Pull    | Push
            # Happy    1    |   0
            #       ---------------
            # Angry    0    |   0
            # This implies that the only congruent motion is when we pull happy. Which is not the case, 
            # so we need a situation where the diagonals are equal to one.
            # Consider, now the following case:
            #       Pull = 1, Push = -1 and Happy = 1, Angry = -1
            #       Pull    | Push
            # Happy    1    |   -1
            #       ---------------
            # Angry    -1   |   1
            # This reflects what we are testing, which is the difference (if any) between congruent
            # and incongruent motions
            DF['is_{}'.format(Target)] = DF['is_{}'.format(Target)].astype(float) - 0.5
            DF['is_pull'] = DF['is_pull'].astype(float) - 0.5

            # Prepare the dependent variable columns
            # Reaction Time
            DF['invRT'] = 1000/DF[self.constants['RT_COLUMN']].astype(float)
            # Mean center
            DF['invRT'] = DF['invRT'] - DF['invRT'].mean()
            # Check for any NaNs, if there are any, then remove them.
            # While we have removed NaNs before, there are cases where there is a 'response' but 
            # upon looking at the acceleration trace we see that this occurs at the very end
            # of the time interval, and seems to be a faulty sensor reading (e.g. exponential 
            # acceleration increase). Hence, there are no 'peaks' after the reaction time
            # threshold has been passed
            NaNIdx = np.where(np.isnan(DF['invRT']))[0]

            # (Change in) Acceleration
            if self.constants['DELTA_A_COLUMN'] in DF.columns:  
                DF['mda'] = DF[self.constants['DELTA_A_COLUMN']].astype(float)
                # Mean center
                DF['mda'] = DF['mda'] - DF['mda'].mean()
                # Check for NaNs
                NaNIdxMDA = np.where(np.isnan(DF['mda']))[0]
                NaNIdx = np.hstack((NaNIdx, NaNIdxMDA))
            # (Change in) Distance
            if self.constants['DELTA_D_COLUMN'] in DF.columns:
                DF['mdd'] = DF[self.constants['DELTA_D_COLUMN']].astype(float)
                # Mean center
                DF['mdd'] = DF['mdd'] - DF['mdd'].mean()
                # Check for NaNs
                NaNIdxMDD = np.where(np.isnan(DF['mdd']))[0]
                NaNIdx = np.hstack((NaNIdx, NaNIdxMDD))

            NaNIdx = np.unique(NaNIdx)

            # Since the DF index does not necessarily correspond to the index found by 
            # np.where (since the DF index can be anything, while np.where returns the
            # integer index), we will create an array of booleans with length equal to 
            # that of the DF. We will replace the indices with NaN values with False
            # and use this array to remove these rows from the DF. 
            BoolIdx = np.ones(len(DF.index), dtype=bool)
            BoolIdx[NaNIdx] = False
        
            print('\n[WARNING] Removing data from {} trials ({}% of total trials) due to missing information'.format(len(NaNIdx), len(NaNIdx)/len(DF.index)*100))

            if sgParams is not None:
                sgParams['OUTPUT'].print(f'[ INFO ] Removing data from {len(NaNIdx)} trials ({len(NaNIdx)/len(DF.index)*100}% of total trials) due to missing data', text_color = 'blue')

            DF = DF.iloc[BoolIdx]

            # Linear Mixed Models 
            # Reaction time
            print('\n')
            print('[INFO] Linear Mixed Model: Reaction Time (1/RT)')
            if sgParams is not None:
                sgParams['OUTPUT'].print('\n[ INFO ] Linear Mixed Model: Reaction Time (1/RT)')
            md = smf.mixedlm(formula = 'invRT ~ is_pull * is_{}'.format(Target), data = DF, 
                            groups=DF['participant'].astype('category'), 
                            re_formula='~is_pull * is_{}'.format(Target))
            mdf = md.fit()
            print(mdf.summary())
            if sgParams is not None:
                sgParams['OUTPUT'].print(f'{mdf.summary()}')
                sgParams['WINDOW'].refresh()
            print('\n')

            if ReturnModel:
                models.update({'rt':md})

            if Save2CSV:
                filename = 'LMM_Results_ReactionTime.csv'
                print('[INFO] Saving (Reaction Time) model results to file {}')
                mdf.summary().tables[1].to_csv('{}\\{}'.format(SavePath, filename))

            # (Change in) Acceleration
            if self.constants['DELTA_A_COLUMN'] in DF.columns:  
                print('[INFO] Linear Mixed Model: (Change in) Acceleration')
                if sgParams is not None:
                    sgParams['OUTPUT'].print('\n[ INFO ] Linear Mixed Model: (Peak) Acceleration')  
                md = smf.mixedlm(formula = 'mda ~ is_pull * is_{}'.format(Target), data = DF, 
                                groups=DF[self.constants['PARTICIPANT_COLUMN']].astype('category'), 
                                re_formula='~is_pull * is_{}'.format(Target))
                mdf = md.fit()
                print(mdf.summary())
                if sgParams is not None:
                    sgParams['OUTPUT'].print(f'{mdf.summary()}')
                    sgParams['WINDOW'].refresh()                
                print('\n')

                if ReturnModel:
                    models.update({'mda':md})

                if Save2CSV:
                    filename = 'LMM_Results_Acceleration.csv'
                    print('[INFO] Saving (acceleration) model results to file {}')
                    mdf.summary().tables[1].to_csv('{}\\{}'.format(SavePath, filename))

            # (Change in) Distance
            if self.constants['DELTA_D_COLUMN'] in DF.columns: 
                print('[INFO] Linear Mixed Model: (Change in) Distance')   
                if sgParams is not None:
                    sgParams['OUTPUT'].print('\n[ INFO ] Linear Mixed Model: (Peak) Distance')                
                md = smf.mixedlm(formula = 'mdd ~ is_pull * is_{}'.format(Target), data = DF, 
                                groups=DF[self.constants['PARTICIPANT_COLUMN']].astype('category'), 
                                re_formula='~is_pull * is_{}'.format(Target))
                mdf = md.fit()
                print(mdf.summary())
                if sgParams is not None:
                    sgParams['OUTPUT'].print(f'{mdf.summary()}')
                    sgParams['WINDOW'].refresh()                
                print('\n')

                if ReturnModel:
                    models.update({'mdd':md})

                if Save2CSV:
                    filename = 'LMM_Results_Distance.csv'
                    print('[INFO] Saving (distance) model results to file {}')
                    mdf.summary().tables[1].to_csv('{}\\{}'.format(SavePath, filename))


        else:
            print('[ERROR] - Expected columns not found for linear mixed models, please check inputted DataFrame')
            if Verbose:
                print('\t Expected ALL of the following columns: {}'.format(ExpectedCols))

        if ReturnModel:
            DF = (DF, models)

        return DF



    def RestructureAvgDF(self, AveragedDF, sgParams = None):

        def FillColData(PID, DF, Movement, Stimulus, Metrics):
            row = {self.constants['PARTICIPANT_COLUMN']:PID,
                   self.constants['CORRECT_RESPONSE_COLUMN']:Movement,
                   self.constants['STIMULUS_SET_COLUMN']:Stimulus
                   }

            for metric in Metrics.keys():
                try:
                    col = '{} {} {}'.format(metric, Movement, Stimulus)
                    row.update({Metrics[metric]:DF.loc[PID, col]})
                except KeyError:
                    pass

            return row

        DF = AveragedDF.copy()
        DF.set_index(['PID'])

        if self.INFO:
            indices = tqdm(DF.index)
        else:
            indices = DF.index

        if sgParams is not None:
            sgParams['OUTPUT'].print('[ INFO ] Restructuring (within participant averaged) dataframe for LMM compatibility...')

        metrics = {'RT':self.constants['RT_COLUMN'], 'DeltaA':self.constants['DELTA_A_COLUMN'], 'DeltaD':self.constants['DELTA_D_COLUMN']}

        RestructuredDF = []

        execution_time = 0
        sTime = time.time() 
        for idx, pid in enumerate(indices):
            if sgParams:
                # # Update progress bar, if user hits cancel, then break the iteration loop
                sgParams['PERCENTAGE'].Update('{:2d} %'.format(int(100*(idx+1)/DF.shape[0])))
                sgParams['PROG_BAR'].UpdateBar(idx + 1, DF.shape[0])
                timeRemainingS = execution_time/(idx + 1) * (DF.shape[0] - idx + 1)
                sgParams['TIME'].Update('Time remaining:{:6d} [s]'.format(int(timeRemainingS)))
                events, values = sgParams['WINDOW'].read(timeout=1)
                if events in (sg.WIN_CLOSED, '-CANCEL-'):
                    sgParams.update({'IS_CLOSED':True})
                    break
            RestructuredDF.append(FillColData(pid, DF, 'Pull', self.Control, metrics))
            RestructuredDF.append(FillColData(pid, DF, 'Pull', self.Target, metrics))
            RestructuredDF.append(FillColData(pid, DF, 'Push', self.Control, metrics))
            RestructuredDF.append(FillColData(pid, DF, 'Push', self.Target, metrics))

            execution_time = time.time() - sTime

        RestructuredDF = pd.DataFrame(RestructuredDF)

        return RestructuredDF



    def AverageBetweenParticipant(self, DFWithinParticipant, sgParams = None):
        Control = self.Control
        Target = self.Target
        
        DF = DFWithinParticipant.copy(deep = True)

        if self.INFO:
            print('[INFO] Averaging results between participants')

        if sgParams is not None:
            sgParams['OUTPUT'].print('[ INFO ] Averaging results between participants...')

        try:
            _dummy = DF['Acc Pull {}'.format(Control)]
            _dummy = DF['Acc Pull {}'.format(Target)]
        except KeyError:
            print('[WARNING] in function <AverageBetweenParticipant> - Inputted DataFrame has not yet been averaged within participants')
            print('\t Attempting to average within participants, calling <AverageWithinParticipant>...')
            DF = self.AverageWithinParticipant(DFWithinParticipant)
        
        DF = DF.set_index(['PID'])
        
        if self.INFO:
            iterations = tqdm(DF.index)
        else:
            iterations = DF.index

        N = len(DF)
        times = DF['time']

        maxT = np.max(times.map(np.max))
        minT = np.min(times.map(np.min))

        dt = 1
        unifiedTime = np.arange(minT, maxT + dt, dt)
        unifiedTime = unifiedTime.reshape((1, len(unifiedTime)))
        dataHost = unifiedTime.copy() * 0
        dataHost = dataHost.astype(float)
        
        FactorMap = {}

        # Find indices of missing data
        MissingIdx = np.array(np.where(DF.isnull().to_numpy())).T
        MissingData = np.sum(DF.isnull())
        for idx, col in enumerate(DF.columns):
            FactorMap.update({col:(1/(N - MissingData[idx]))})

        # Create dictionary to store data
        data = {}
        for cIDX, col in enumerate(DF.columns):
            pIDX = 0
            Building = True
            while Building:
                if pIDX == DF.shape[0]:
                    Building = False
                    data.update({col:np.nan})
                if pIDX in MissingIdx[:, 0] and cIDX in MissingIdx[:, 1]:
                    pIDX += 1
                else:
                    Building = False
                    try:
                        shape = DF.iloc[pIDX, cIDX].shape
                        if len(shape) == 0:
                            data.update({col:DF.iloc[pIDX, cIDX] * 0})
                        else:
                            if len(shape) == 1:
                                shape = (1, shape[0])
                            tup = shape[0] * [dataHost.copy()]
                            colData = np.vstack(tuple(tup))
                            data.update({col:colData})
                    except AttributeError:
                        data.update({col:DF.iloc[pIDX, cIDX] * 0})

        execution_time = 0
        sTime = time.time()
        for pIdx, pid in enumerate(iterations):
            if sgParams:
                # # Update progress bar, if user hits cancel, then break the iteration loop
                # if not sg.OneLineProgressMeter('Filtering data...', i + 1, len(files), key='-IMPORT_PROGRESS-', orientation='h'):
                #     break
                sgParams['PERCENTAGE'].Update('{:2d} %'.format(int(100*(pIdx+1)/DF.shape[0])))
                sgParams['PROG_BAR'].UpdateBar(pIdx + 1, DF.shape[0])
                timeRemainingS = execution_time/(pIdx + 1) * (DF.shape[0] - pIdx + 1)
                sgParams['TIME'].Update('Time remaining:{:6d} [s]'.format(int(timeRemainingS)))
                events, values = sgParams['WINDOW'].read(timeout=1)
                if events in (sg.WIN_CLOSED, '-CANCEL-'):
                    sgParams.update({'IS_CLOSED':True})
                    break
            pData = DF.loc[pid, :]
            pTime = pData['time']
            try:
                pStartIdx = np.where(unifiedTime[0] >= pTime[0])[0][0]
                pEndIdx = np.where(unifiedTime[0] >= pTime[-1])[0][0] + 1
            # If this error arises, it means that there is no time data (participant only
            # completed practice trials)
            # TODO: Flag for removal
            except TypeError:
                pass
            
            for cIdx, col in enumerate(pData.index):
                try:
                    # We are dealing with values (e.g. RT)
                    if len(pData[col].shape) == 0:
                        surrogateData = pData[col] * FactorMap[col]
                        ax = None

                    # # Then we are dealing with a 1-d array (time)
                    elif col == 'time':
                        surrogateData = unifiedTime.copy() * FactorMap[col]
                        ax = 1
                    # Otherwise, we need to loop through entries of n-d array
                    else:
                        n = len(pData[col])
                        # Create n copies of surrogateData to be filled in
                        tup = []
                        for row in range(pData[col].shape[0]):
                            tup.append(dataHost.copy())
                        surrogateData = np.vstack(tuple(tup))
                        surrogateData[:, pStartIdx:pEndIdx] = pData[col] * FactorMap[col]
                        ax = 1

                # When value error is raised, there is no data for that column, so leave data empty
                except AttributeError:
                    surrogateData = data[col]
                
                if pIdx in MissingIdx[:, 0] and cIdx in MissingIdx[:, 1]:
                    pass
                else:
                    oldData = data[col]
                    if ax:
                        newData = oldData.copy()
                        for row in range(surrogateData.shape[0]):
                            newData[row] = np.nansum(np.vstack((oldData[row], surrogateData[row])), axis = (ax-1))
                    else:
                        if not np.isnan(pData[col]):
                            newData = np.nansum(np.vstack((oldData, surrogateData)))
                    data.update({col : newData})

            execution_time = time.time() - sTime


        data.update({'PID' : 'Average'})
        AvgDF = pd.DataFrame(data=[data.values()], columns=data.keys())

        return AvgDF



    def AverageWithinParticipant(self, DataFrame, sgParams = None):
        Control = self.Control
        Target = self.Target

        DF = DataFrame.copy(deep = True)
        DFByPID = DF.set_index([self.constants['PARTICIPANT_COLUMN']])
        UniquePIDs = np.unique(DF[self.constants['PARTICIPANT_COLUMN']])

        if self.INFO:
            print('[INFO] Averaging results within participants...')
            iterations = tqdm(UniquePIDs)
        else:
            iterations = UniquePIDs

        if sgParams is not None:
            sgParams['OUTPUT'].print('[ INFO ] Averaging results within participants...')

        NumParticipants = len(UniquePIDs)

        # Add the two metrics which are always present
        metrics = {'RT' : np.nan,
                   'Acc' : [np.array(np.nan)]*NumParticipants}

        # Add 'optional' metrics; parameters which can be optionally
        # computed by the user
        Distances = [self.constants['DISTANCE_X_COLUMN'], self.constants['DISTANCE_Y_COLUMN'], self.constants['DISTANCE_Z_COLUMN']]
        if all(i in DF.columns for i in Distances):
            metrics.update({'Dist' : [np.array(np.nan)]*NumParticipants})

        if self.constants['DELTA_A_COLUMN'] in DF.columns:
            metrics.update({'DeltaA' : np.nan})

        if self.constants['DELTA_D_COLUMN'] in DF.columns:
            metrics.update({'DeltaD' : np.nan})

        movements = ['Push', 'Pull']

        # Generate overview columns
        Cols = {'PID' : ' ',
                'time' : [np.array(np.nan)]*NumParticipants}
        for mov in movements:
            for m in metrics.keys():
                Cols.update({'{} {} {}'.format(m, mov, Control) : metrics[m]})
                Cols.update({'{} {} {}'.format(m, mov, Target) : metrics[m]})

        Data = []

        execution_time = 0
        sTime = time.time()
        for idx, pid in enumerate(iterations):
            if sgParams:
                # # Update progress bar, if user hits cancel, then break the iteration loop
                # if not sg.OneLineProgressMeter('Filtering data...', i + 1, len(files), key='-IMPORT_PROGRESS-', orientation='h'):
                #     break
                sgParams['PERCENTAGE'].Update('{:2d} %'.format(int(100*(idx+1)/len(UniquePIDs))))
                sgParams['PROG_BAR'].UpdateBar(idx + 1, len(UniquePIDs))
                timeRemainingS = execution_time/(idx + 1) * (len(UniquePIDs) - idx + 1)
                sgParams['TIME'].Update('Time remaining:{:6d} [s]'.format(int(timeRemainingS)))
                events, values = sgParams['WINDOW'].read(timeout=1)
                if events in (sg.WIN_CLOSED, '-CANCEL-'):
                    sgParams.update({'IS_CLOSED':True})
                    break

            ParticipantData = DFByPID.loc[pid, :]
            N = len(ParticipantData)
            PracticeIdx = np.where(DFByPID.loc[pid, 'is_practice'])[0]
            AllIdx = np.arange(0, N, 1)
            AllIdx = np.delete(AllIdx, PracticeIdx)
            ParticipantData = ParticipantData.iloc[AllIdx, :]
            times = ParticipantData[self.constants['TIME_COLUMN']]
            maxT = np.max(times.map(max))
            minT = np.min(times.map(min))

            # Some participants didn't even get past the practice trials
            if len(AllIdx) > 0:
                dt = 1
                UnifiedTime = np.arange(minT, maxT + dt, dt)
                
                Data.append(Cols.copy())
                Data[idx]['PID'] = pid
                Data[idx]['time'] = UnifiedTime

                for mov in movements:
                    C_Avg = self._AverageOverCondition(ParticipantData, UnifiedTime, mov, Control)
                    T_Avg = self._AverageOverCondition(ParticipantData, UnifiedTime, mov, Target)
                    for m in metrics:
                        Data[idx]['{} {} {}'.format(m, mov, Control)] = C_Avg[m]
                        Data[idx]['{} {} {}'.format(m, mov, Target)] = T_Avg[m]

            else:
                Data.append(Cols.copy())
                for col in Cols.keys():
                    Data[-1][col] = np.nan
                Data[-1]['PID'] = pid
            
            execution_time = time.time() - sTime
        
        OutDF = pd.DataFrame(Data)

        return OutDF



    def _AverageOverCondition(self, participantData, unifiedTime, movement, stimulus):
        movementIdx = np.where(participantData[self.constants['CORRECT_RESPONSE_COLUMN']] == movement)[0]
        stimulusIdx = np.where(participantData[self.constants['STIMULUS_SET_COLUMN']] == stimulus)[0]
        conditionIdx = np.intersect1d(movementIdx, stimulusIdx)

        condData = participantData.iloc[conditionIdx, :]
        N = len(condData)

        distCols = [self.constants['DISTANCE_X_COLUMN'], self.constants['DISTANCE_Y_COLUMN'], self.constants['DISTANCE_Z_COLUMN']]
        isDist = all(col in condData.columns for col in distCols)
        isDeltaA = self.constants['DELTA_A_COLUMN'] in condData.columns
        isDeltaD = self.constants['DELTA_D_COLUMN'] in condData.columns

        if N > 0:
            times = condData[self.constants['TIME_COLUMN']]

            Acc = np.zeros((3, N, len(unifiedTime)))
            if isDist:
                Dist = np.zeros((3, N, len(unifiedTime)))

            for i in range(N):
                startIdx = np.where(unifiedTime >= times[i][0])[0][0]
                endIdx = np.where(unifiedTime >= times[i][-1])[0][0] + 1

                iData = condData.iloc[i, :]

                Acc[0, i, startIdx:endIdx] = iData[self.constants['ACCELERATION_X_COLUMN']]
                Acc[1, i, startIdx:endIdx] = iData[self.constants['ACCELERATION_Y_COLUMN']]
                Acc[2, i, startIdx:endIdx] = iData[self.constants['ACCELERATION_COLUMN']]

                if isDist:
                    Dist[0, i, startIdx:endIdx] = iData[self.constants['DISTANCE_X_COLUMN']]
                    Dist[1, i, startIdx:endIdx] = iData[self.constants['DISTANCE_Y_COLUMN']]
                    Dist[2, i, startIdx:endIdx] = iData[self.constants['DISTANCE_Z_COLUMN']]

            # Runtime warnings may appear due to this averaging, since there could instances where
            # we are averaging arrays of just nans. 
            RTs = condData[self.constants['RT_COLUMN']]
            AvgRT = np.nanmean(RTs.to_numpy().astype(float))

            AvgAcc = np.nanmean(Acc, axis = 1)
            if isDist:
                AvgDist = np.nanmean(Dist, axis = 1)

            if isDeltaA:
                DAs = condData[self.constants['DELTA_A_COLUMN']]
                AvgDA = np.nanmean(DAs.to_numpy().astype(float))

            if isDeltaD:
                DDs = condData[self.constants['DELTA_D_COLUMN']]
                AvgDD = np.nanmean(DDs.to_numpy().astype(float))

        # Some participants did not complete all conditions 
        else:
            AvgRT = np.nan   
            AvgAcc = np.nan
            if isDist:
                AvgDist = np.nan
            if isDeltaA:
                AvgDA = np.nan
            if isDeltaD:
                AvgDD = np.nan

        Vals = {'RT':AvgRT,
                'Acc':AvgAcc}
        if isDist:
            Vals.update({'Dist':AvgDist})
        if isDeltaA:
            Vals.update({'DeltaA':AvgDA})
        if isDeltaD:
            Vals.update({'DeltaD':AvgDD})

        return Vals