# Class to handle importing, filtering, and pre-processing of data from the AAT

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

class DataHandler:

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
                        'GYRO_NOISE' : 'gyro_noise'}
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



    def CheckFiles(self, files, path):
        os.chdir(path)
        output = all(os.path.isfile(os.path.join(path, file)) for file in files)
        os.chdir(self.cwd)
        return output

    
    
    def SaveDF(self, filename, dataframe, path):
        os.chdir(path)
        dataframe.to_pickle('./{}'.format(filename))
        os.chdir(self.cwd)



    def LoadDF(self, filename, path):
        os.chdir(path)
        df = pd.read_pickle(os.path.join(path, filename))
        self.Data = df
        os.chdir(self.cwd)
        return df



    def RenameDFCols(self):
        # Add something to rename acceleration column to acceleration_z
        pass


    # Obtaining distance is not so trivial -> Due to double integration, noise and bias cause large errors in estimates
    # Issue with double integrations is that we integrate sensor noise and it explodes the distance estimates
    # -> Try to mitigate this by not taking straight integrals but instead using Physical principles
    # https://www.youtube.com/watch?v=C7JQ7Rpwn2k @ 25:22 shows that, in 1 second, we can have drift of up to 20 cm
    # -> Errors in orientation estimatation are an order of magnitude worse -> 1 degree off could lead to several 
    #    meters of error in distance. https://www.cl.cam.ac.uk/techreports/UCAM-CL-TR-696.pdf
    # -> Corrections to gyroscopic data can also be used with the gravity vector is known 
    # Perhaps we can dynamics of system -> use kalman filter to estimate distance 
    # Use sensor fusion to get better state estimates -> Need additional data (e.g. Magnetometer; then we can implement
    # a madgwick filter)
    def ComputeDistanceVelocity(self):

        pass



    def CheckInfo(self, ExpectedInfoList, GivenInfoDict):
        if all(info in GivenInfoDict for info in ExpectedInfoList):
            output = True
        else:
            output = False

        return output


    # Only accept array like or list objects
    def HasArrayData(self, DataRow, Column):
        try:
            # If we can obtain a shape from the data an array-like type
            if DataRow[self.constants[Column]].shape[0] > 0:
                HasData = True
            else:
                HasData = False
        except AttributeError:
            try: 
                # Try to call a list attribute, if DataFrame[Column] is
                # not a list, it will raise an AttributeError
                DataRow[self.constants[Column]].count(0)
                if len(DataRow[self.constants[Column]]) > 0:
                    HasData = True
                else:
                    HasData = False
            except AttributeError:
                HasData = False
        return HasData



    def FilterNaNs(self, DataFrame):
        DF = DataFrame.copy(deep = True)
        if self.INFO:
            print("[INFO] Removing NaN Values...")
            iterrows = tqdm(DF.iterrows(), total=DF.shape[0])
        else:
            iterrows = DF.iterrows()

        N = DF.shape[0]
        Indices_to_remove_NaN = []

        for i, row in iterrows:
            # Remove missing data
            for col in self.DataCols2Check:
                if not self.HasArrayData(row, col) and i not in Indices_to_remove_NaN:
                    Indices_to_remove_NaN.append(i)

        if self.INFO:
            print('[INFO] Filtering results:')
            print('[INFO] \t Percentage of data which is missing: {}'.format(len(Indices_to_remove_NaN)/N*100))

        DF = DF.drop(index = Indices_to_remove_NaN)
        DF = DF.reset_index(drop = True)

        return DF


    # Compute RT can be done before orientation corrections and acceleration calibration
    # because we are looking for a reaction - which will be a change in acceleration. This
    # is therefore independent of the absolute of acceleration (only relative values are 
    # considered). Moreover, the orientation changes are given with respect to the initial
    # orientation (since, with current information, precise orientation w.r.t gravity is 
    # unknown). Hence, it is assumed that the change in angles is ~0 before a reaction. 
    #   Inputs:     RTResLB = Acceleration Lowerbound to be considered a response
    #               RTHeightThresRatio = Ratio of maximum acceleration 
    #                   In this case we take max(RTResLB, RTHeightThresRatio) as the minimum
    #                   peak height
    #               RTPeakDistance = Minimum allowed distance between peaks (in milliseconds)
    def ComputeRT(self, DataFrame, RTResLB = 1, RTHeightThresRatio = 0.5, RTPeakDistance = 10):
        DF = DataFrame.copy(deep = True)
        if self.INFO:
            print("[INFO] Computing Reaction Times...")
            iterrows = tqdm(DF.iterrows(), total=DF.shape[0])
        else:
            iterrows = DF.iterrows()

        N = DF.shape[0]
        RTs = pd.Series(index = np.arange(0, N), name = self.constants['RT_COLUMN'])

        for i, row in iterrows:
            # Some signals have a lot of jitter at the beginning with relatively high accelerations. To reduce the number of participants
            # falsely identified as having too fast reaction times, we will consider any peak larger than half of the highest peak as a 
            # significant peak. 
            RTHeightThres = max(max(abs(row[self.constants['ACCELERATION_COLUMN']]))*RTHeightThresRatio, RTResLB)
            # First find all peaks in abs(signal)
            all_peaks = spsig.find_peaks(abs(row[self.constants['ACCELERATION_COLUMN']]), distance = RTPeakDistance, height = RTHeightThres)
            if len(all_peaks[0]):
                # Use these all peaks to find the peak prominences: We expect that the main response will have the highest prominence values
                prominences = spsig.peak_prominences(abs(row[self.constants['ACCELERATION_COLUMN']]), all_peaks[0])[0]
                # Given the shape of the signal, there should be 2 peaks (one corresponding to the initial direction, and the other returning to 
                # 'origin'). Hence, we will take half of the highest prominence as our threshold. 
                min_prominence = np.sort(prominences[::-1])[0]/2
                peaks = spsig.find_peaks(abs(row[self.constants['ACCELERATION_COLUMN']]), prominence = min_prominence, distance = RTPeakDistance, height = RTHeightThres)
                initial_peak_idx = peaks[0][0]
                # The final reaction time is given by the time of the peak - the time at the start of the measurement - a constant
                # to account for the fact that the reaction occurs before the peak. 
                RTs[i] = row[self.constants['TIME_COLUMN']][initial_peak_idx] - row[self.constants['TIME_COLUMN']][0] - RTPeakDistance
            else:
                RTs[i] = 'No Reaction'    

        DF[self.constants['RT_COLUMN']] = RTs

        return DF



    def HasRealisticRT(self, DataRow, RTThreshold):
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
    # (assumed) stationary period of time before a reaction occurs. 
    def CalibrateData(self, Row):
        DataRow = Row.copy(deep = True)
        
        a_std = np.nan
        g_std = np.nan

        def Calibrate(Data, column, end_idx):
            x = Data[self.constants[column]][0:end_idx]
            std = np.nanstd(x)
            offset = np.nanmean(x)
            return offset, std

        rt = DataRow[self.constants['RT_COLUMN']]

        time_a = DataRow[self.constants['TIME_COLUMN']]
        try:
            idx_rt_a = np.where(time_a < rt)[0][-1]
        except TypeError:
            idx_rt_a = len(time_a) - 1

        if not idx_rt_a == 0:
            a_cols = ['ACCELERATION_X_COLUMN', 'ACCELERATION_Y_COLUMN', 'ACCELERATION_COLUMN']
            a_std = {}
            for col in a_cols:
                offset, std = Calibrate(DataRow, col, idx_rt_a)
                DataRow[self.constants[col]] -= offset
                key = col.split('_')[1]
                if len(key) > 1:
                    key = 'Z'
                a_std.update({key:std})


        if len(DataRow[self.constants['GYRO_TIME_COLUMN']]) > 0:
            time_g = DataRow[self.constants['GYRO_TIME_COLUMN']]
            try:
                idx_rt_g = np.where(time_g < rt)[0][-1]
            except TypeError:
                idx_rt_g = len(time_g) - 1

            if not idx_rt_g == 0:
                g_cols = ['GYRO_X_COLUMN', 'GYRO_Y_COLUMN', 'GYRO_Z_COLUMN']
                g_std = {}
                for col in g_cols:
                    offset, std = Calibrate(DataRow, col, idx_rt_g)
                    DataRow[self.constants[col]] -= offset
                    key = col.split('_')[1]
                    g_std.update({key:std})

        stds = [a_std, g_std]
        
        return DataRow, stds



    def HasRealisticAccelerations(self, DataRow, MaxArmDist, TimeScale = 1000, Tolerance = 0.10):
        MaxDist = MaxArmDist[DataRow['gender']]
        # Add catch for any missing data (i.e. np.nans)
        try:
            delta_t = (DataRow[self.constants['TIME_COLUMN']][-1] - DataRow[self.constants['TIME_COLUMN']][0])/TimeScale
            a_limit = 2*MaxDist/(delta_t**2)

            ax = DataRow[self.constants['ACCELERATION_X_COLUMN']]
            ay = DataRow[self.constants['ACCELERATION_Y_COLUMN']]
            az = DataRow[self.constants['ACCELERATION_COLUMN']]

            a_tot = np.sqrt((np.nanmean(ax)**2 + np.nanmean(ay)**2 + np.nanmean(az)**2))

            if a_tot < (1+Tolerance)*a_limit:
                AccIsRealistic = True
            else:
                AccIsRealistic = False

        except TypeError:
            AccIsRealistic = False
        
        return AccIsRealistic


    # Average human reaction time to visual stimulus is 0.25 seconds -> set default threshold to 200
    # Human arm lengths https://www.researchgate.net/figure/ARM-LENGTH-IN-MM-BY-SEX-AND-AGE_tbl3_10567860
    def FilterData(self, Data, RemoveRT = True, KeepNoRT = False, RTThreshold = 200, CalibrateAcc = True, RemoveAcc = True, MaxArmDist = {'female':0.735, 'male':0.810}):
        DF = Data.copy(deep = True)

        if self.INFO:
            print("[INFO] Running Filtering...")
            iterrows = tqdm(DF.iterrows(), total=DF.shape[0])
        else:
            iterrows = DF.iterrows()

        N = DF.shape[0]
        Indices_to_remove_RT = []
        Indices_to_remove_Acc = []
        NoReaction_idx = []
        AccelerationNoiseList = []
        GyroNoiseList = []
        CalibratedList = []

        try:
            if DF['rt'][0] > 0:
                pass
        except KeyError:
            print('[WARNING] The inputted DataFrame for <FilterData> does not have a reaction time column.\n\tI will try to compute reaction times here.\n\tNOTE: Default values are being used.')
            DF = self.ComputeRT(DF)

        for i, row in iterrows:
            # Remove implausible reaction times 
            if RemoveRT:
                HasRealisticRT, NoReaction = self.HasRealisticRT(row, RTThreshold)
                if NoReaction:
                    NoReaction_idx.append(i)
                if not HasRealisticRT and i not in NoReaction_idx:
                    Indices_to_remove_RT.append(i)

            # Correct accelerations for offsets and extract noise characteristics
            if CalibrateAcc:
                if i not in Indices_to_remove_RT:
                    CalibratedData, [a_noise, g_noise] = self.CalibrateData(row)
                else:
                    CalibratedData = DF.loc[i, :].copy(deep = True)
                    a_noise, g_noise = np.nan, np.nan
                CalibratedList.append(CalibratedData)
                AccelerationNoiseList.append(a_noise)
                GyroNoiseList.append(g_noise)
            
            # Remove implausible accelerations (either due to sensor errors or otherwise)
            if RemoveAcc:
                # Need to check if there is the necessary info in the given row. 
                if not self.HasRealisticAccelerations(row, MaxArmDist):
                    Indices_to_remove_Acc.append(i)
        if KeepNoRT:
            Indices_to_remove = np.unique(Indices_to_remove_RT + Indices_to_remove_Acc)
        else:
            Indices_to_remove = np.unique(Indices_to_remove_RT + Indices_to_remove_Acc + NoReaction_idx)
    
        if CalibrateAcc:
            DF = pd.concat(CalibratedList, axis = 1).transpose()
            DF[self.constants['ACCELEROMETER_NOISE']] = AccelerationNoiseList
            DF[self.constants['GYRO_NOISE']] = GyroNoiseList

        if self.INFO:
            print('[INFO] Filtering results:')
            if RemoveRT:
                print('[INFO] \t Percentage of data with implausible (<{} ms) reaction times: {}'.format(RTThreshold, len(Indices_to_remove_RT)/N*100))
                print('[INFO] \t Percentage of data with No Reaction: {}'.format(len(NoReaction_idx)/N*100))
            if RemoveAcc:
                print('[INFO] \t Percentage of data with implausible accelerations: {}'.format(len(Indices_to_remove_Acc)/N*100))
            print('[INFO] Total percentage of data filetered (Keep No Reactions = {}): {}'.format(KeepNoRT, len(Indices_to_remove)/N*100))
            
        Removed_Data = DF[DF.index.isin(Indices_to_remove)]
        DF = DF.drop(index = Indices_to_remove)
        DF = DF.reset_index(drop = True)

        return DF, Removed_Data


    # Integrate gyroscopic data (angular rates) to get angles
    # Ideally, we would use some from of state estimation (e.g. Kalman filter) to
    # obtain a better representation of the true angular rates (since integrating
    # gyroscopic data incurs significant errors, if drift and biases are not
    # accounted for). However, there is insufficient information to adequately do
    # this at this point. The time scales we are looking at are small 
    # (~ 2 seconds), so errors will not have propagated so much. That said, 
    # depending on the sensor, errors can be in the order of centimeters after even
    # just a second. 
    def IntegrateGyro(self, time, g_vec, theta_ic, scale = 1000):

        th_vec = np.zeros((3, len(time))) + theta_ic

        for t in range(len(time) - 1):
            dt = (time[t + 1] - time[t])/scale

            # Forward Euler Integration
            for i in range(len(th_vec)):
                th_vec[i, t + 1] = th_vec[i, t] + g_vec[i, t] * dt
        
        return th_vec


    # Get quaternion representation of the angle vector
    def GetQuat(self, theta_vec):
        quat_vec = np.zeros((4, len(theta_vec[0])))

        for t in range(len(theta_vec[0])):
            mag = np.sqrt( (theta_vec[0, t]**2 + theta_vec[1, t]**2 + theta_vec[2, t]**2) )

            # Normalized angle vector
            Nth_vec = theta_vec[:, t]/mag

            thetaOver2 = mag/2
            sinTO2 = np.sin(thetaOver2)
            cosTO2 = np.cos(thetaOver2)

            quat_vec[0][t] = cosTO2
            quat_vec[1][t] = sinTO2 * Nth_vec[0]
            quat_vec[2][t] = sinTO2 * Nth_vec[1]
            quat_vec[3][t] = sinTO2 * Nth_vec[2]

        return quat_vec



    def Correction4Rotations(self, a_vec, th_vec, AngleThreshold = 1*np.pi/180):
        # As order of rotations is important, infer the order of rotations
        # based on which axes first exceed the 'AngleThreshold' 
        # The default angle threshold is set to 1 degree, as below this the small
        # angle approximation typically holds (i.e. rotations are negligible)
        IndexExceedingThres =  [(np.where(abs(th_vec[0]) > AngleThreshold)[0][0]),
                                (np.where(abs(th_vec[1]) > AngleThreshold)[0][0]),
                                (np.where(abs(th_vec[2]) > AngleThreshold)[0][0])]
        # We reverse the order of the argsort since argsort will give the highest
        # value first, whereas we want the lowest values (i.e. first instance)
        RotOrder = np.argsort(IndexExceedingThres)[::-1]

        quat_vec_original = self.GetQuat(th_vec)
        # We need to re-arrange the quaternion vector to correspond to the rotation
        # vector
        quat_vec = np.copy(quat_vec_original)
        for i in range(len(RotOrder)):
            quat_vec[i + 1] = quat_vec_original[RotOrder[i] + 1]
        
        # Since we are going from the body, to the inertial frame, we need to take the inverse of the
        # quaternion, q (which is equivalent to its conjugate). We are going from the body frame to the
        # interial frame since we the accelerations are measured w.r.t to the device (i.e. body frame)
        # hence, to account for any rotations (w.r.t the initial position, which we interpret as the 
        # inertial frame) we need to project our accelerations from the body frame to the inertial frame.
        q0 = quat_vec[0]
        q1 = quat_vec[1]
        q2 = quat_vec[2]
        q3 = quat_vec[3]

        # Rotation Matrix
        R_1 = [(q0*q0 + q1*q1 - q2*q2 -q3*q3), (2*(q1*q2 - q0*q3)), (2*(q0*q2 + q1*q3))]
        R_2 = [(2*(q1*q2 + q0*q3)), (q0*q0 - q1*q1 + q2*q2 - q3*q3), (2*(q2*q3 - q0*q1))]
        R_3 = [(2*(q1*q3 - q0*q2)), (2*(q0*q1 + q2*q3)), (q0*q0 - q1*q1 - q2*q2 + q3*q3)]

        R = np.mat(np.vstack((R_1, R_2, R_3)))

        return np.reshape(R*a_vec, (3,))

    
    def ResampleData(self, DataFrame):
        DF = DataFrame.copy(deep = True)
        ResampledList = []
        if self.INFO:
            print("[INFO] Running Resampling...")
            iterrows = tqdm(DF.iterrows(), total=DF.shape[0])
        else:
            iterrows = DF.iterrows()
        for i, row in iterrows:
            _ResampledRow = self.ResampleRow(row)
            ResampledList.append(_ResampledRow)

        ResampledDF = pd.concat(ResampledList, axis = 1).transpose()
        return ResampledDF



    def ResampleRow(self, data_row):
        self.Data2Resample = {'acceleration_z':'ACCELERATION_COLUMN',
                              'acceleration_x':'ACCELERATION_X_COLUMN',
                              'acceleration_y':'ACCELERATION_Y_COLUMN',
                              'gyro_x':'GYRO_X_COLUMN',
                              'gyro_y':'GYRO_Y_COLUMN',
                              'gyro_z':'GYRO_Z_COLUMN'}
        ResampledRow = data_row.copy(deep = True)
        ResampledTime = self.AlignTimeArrays(data_row[self.constants['TIME_COLUMN']], data_row[self.constants['GYRO_TIME_COLUMN']], dt = 1)
        ResampledRow[self.constants['TIME_COLUMN']] = ResampledTime
        for key, value in self.Data2Resample.items():
            if key.startswith('acceleration'):
                try:
                    ResampledRow[self.constants[value]] = self.Interpolate(data_row[self.constants[value]], data_row[self.constants['TIME_COLUMN']], ResampledTime)
                except TypeError:
                    pass
            elif key.startswith('gyro'):
                try:
                    ResampledRow[self.constants[value]] = self.Interpolate(data_row[self.constants[value]], data_row[self.constants['GYRO_TIME_COLUMN']], ResampledTime)
                except TypeError:
                    pass
            else:
                pass
        #Remove Gyro time column as it is now redundant 
        ResampledRow.drop(self.constants['GYRO_TIME_COLUMN'])
        return ResampledRow
    


    def Interpolate(self, y, x1, x2):
        # Scan input data for nans, remove them from the array. The best we can do is interpolate these points with neighbors
        y_nonan = y[np.invert(np.isnan(y))]
        x1 = x1[np.invert(np.isnan(y))]
        x1_nonan = x1[np.invert(np.isnan(x1))]
        y_nonan = y_nonan[np.invert(np.isnan(x1))]

        Bspline = spinter.splrep(x1_nonan, y_nonan)
        y2 = spinter.splev(x2, Bspline)

        return y2


    # dt is in milliseconds 
    def AlignTimeArrays(self, acc_time, gyro_time, dt = 1):
        # All experiments should have accelerometer data, but in case they do not
        # TODO: Add tags for participants with missing data, can be filtered out
        try:
            if len(acc_time) & len(gyro_time):
                # Some devices (using Simple Accelerometer) do not have gyroscopic data
                t_start = np.nanmax((acc_time[0], gyro_time[0]))
                t_end = np.nanmin((acc_time[-1], gyro_time[-1]))
                time_array = np.arange(t_start, t_end, dt)
            elif len(acc_time):
                t_start = acc_time[0]
                t_end = acc_time[-1]
                time_array = np.arange(t_start, t_end, dt)
            else:
                time_array = np.nan
        except TypeError:
            time_array = np.nan

        return time_array

    
    def ImportData(self):
        self._Data = []
        if self.INFO:
            print('[INFO] Loading Participant Data...')
            files = tqdm(os.listdir(self.data_path))
        else:
            files = os.listdir(self.data_path)
        for f in files:
            if f.endswith(".json"):
                self._participant_file = self.LoadJson(os.path.join(self.data_path, f))
                if self.CheckInfo(['participantId', 'condition'], self._participant_file):
                    self._participant_id = self._participant_file['participantId']
                    self._condition = self._participant_file['condition']
                    self._CatQCols = self.GetCategoricalQuestionCols(self._condition)
                    self._CondTable = self.LoadParticipantData()
                    self._CondTable = self._CondTable.reset_index().set_index(['participant', 'session', 'block', 'trial'])
                    self._CondTable[self._CatQCols] = self._CondTable[self._CatQCols].astype('category')

                    self._Data.append(self._CondTable)

        self.Data = pd.concat(self._Data, sort = True).reset_index()
        return self.Data

    
    # Function used to extract the experiment data from the .json files
    #   Inputs: None; This function works entirely on variables defined within the class,
    #           it calls on other functions in the class which generate necessary variables
    #           However, the necessary constants can only be generated when the class is 
    #           initiallized (i.e. __init__ function called in the main script)
    #   Outputs: Filled in ConditionTable
    def LoadParticipantData(self):
        self._CondTable = self.GetCondTable(self._condition)

        # Loop through the keys in the participant json file. If the keys match the 
        # (assosicated Json keys in) ExpectedParticipantData, then we fill in the 
        # corresponding column in the Condition table
        # The ExpectedParticipantData dictionary maps the jsonKeys to the columns in
        # the CondTable, which are participant dependent but session independent.
        # For example, the participant id. 
        # Session Specific data is not included in the ExpectedParticipantData, and 
        # is therefore handled in the else segment. 
        for jsonKey in self._participant_file.keys():
            if jsonKey in self.ExpectedParticipantData.keys():
                self._CondTable[self.ExpectedParticipantData[jsonKey]] = self._participant_file[jsonKey]
            else:
                self._SessionDict = self._participant_file[jsonKey]
                for task in self._SessionDict.keys():
                    # Possible tasks are AAT or answering questionaires
                    if task == 'AAT':
                        # Start iterable at 1 (No block 0)
                        for block_idx, block in enumerate(self._SessionDict[task]['blocks'], 1):
                            # There are instances where there is a 'null' for a trial (First entry in block)
                            # So we will ignore the first entry
                            for trial_idx, trial in enumerate(block[1:], 1):
                                idx = (jsonKey, block_idx, trial_idx)
                                if idx in self._CondTable.index:
                                    # Store the data from the block and trial into the data frame
                                    # Store data which can be directly obtained from the .json file (i.e. no processing or
                                    # special handling required)
                                    for key in self.ExpectedBlockData.keys():
                                        try:
                                            self._CondTable.at[idx, self.constants[key]] = trial[self.ExpectedBlockData[key]]
                                        except:
                                            self._CondTable.at[idx, self.constants[key]] = np.nan
                                    # Store data which requires special handling or additional processing
                                    # Stimulus set:
                                    try:
                                        self._CondTable.at[idx, self.constants['STIMULUS_SET_COLUMN']] = self.INV_stimulus_sets[trial[self.ExpectedBlockData['STIMULUS_COLUMN']]]
                                    except:
                                        self._CondTable.at[idx, self.constants['STIMULUS_SET_COLUMN']] = np.nan

                                    acc_times = np.array([])
                                    gyro_times = np.array([])

                                    for key, value in self.ExpectedNumericData.items():
                                        if value in trial.keys():
                                            val = pd.Series(trial[self.ExpectedNumericData[key]])
                                            val.index = val.index.astype('int')
                                            val = val.sort_index()
                                            if (key.startswith('acceleration')) & (len(acc_times) == 0):
                                                acc_times = val.index.values
                                                self._CondTable.at[idx, self.constants['TIME_COLUMN']] = acc_times
                                            elif (key.startswith('gyro')) & (len(gyro_times) == 0):
                                                gyro_times = val.index.values
                                                self._CondTable.at[idx, self.constants['GYRO_TIME_COLUMN']] = gyro_times
                                            val = val.values
                                            self._CondTable.at[idx, self.constants[self.ExpectedNumericData2Constants[key]]] = val
                                        else:
                                            if len(acc_times) == 0:
                                                self._CondTable.at[idx, self.constants['TIME_COLUMN']] = np.array([])
                                            elif len(gyro_times) == 0:
                                                self._CondTable.at[idx, self.constants['GYRO_TIME_COLUMN']] = np.array([])
                                            self._CondTable.at[idx, self.constants[self.ExpectedNumericData2Constants[key]]] = np.array([])
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
    def GetCondTable(self, condition):
        try:
            output = self.CondTablesDict['{}'.format(condition)].copy(deep = True)
        except AttributeError:
            self.ImportConditions(self.cond_path)
            output = self.CondTablesDict['{}'.format(condition)].copy(deep = True)

        return output


    # Function used to get the categorical question columns from the Condition Tables
    # of a given condition
    #   Input: condition of interest
    #   Output: Categorical Question Columns (list)
    def GetCategoricalQuestionCols(self, condition):
        try:
            output = self.CategoricalQuestionColsDict['{}'.format(condition)]
        except AttributeError:
            self.ImportConditions(self.cond_path)
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
    def ImportConditions(self, cond_path):
        self.conditions = self.LoadJson(os.path.join(cond_path, "conditions.json"))
        self.sessions = self.LoadJson(os.path.join(cond_path, "sessions.json"))
        self.tasks = self.LoadJson(os.path.join(cond_path, "tasks.json"))
        self.stimulus_sets = self.LoadJson(os.path.join(cond_path, "stimulus_sets.json"))
        self.INV_stimulus_sets = dict((v, k) for k in self.stimulus_sets for v in self.stimulus_sets[k])

        self.CondTablesDict = {}
        self.CategoricalQuestionColsDict = {}

        for condition in self.conditions:
            session_names = self.conditions[condition]['sessions']
            q_cols, numeric_q_cols, cat_q_cols = [], [], []
            sessionlst, blocks, trials, total_trials, practice = [], [], [], [], []
            for session in session_names:
                task_names = self.sessions[session]['tasks']
                for task in task_names:
                    try:
                        task_type = self.tasks[task]['type']
                    except KeyError:
                        try:
                            task = self.tasks[task]['parent']
                            task_type = self.tasks[task]['type']
                        except KeyError:
                            task_type = None
                    if task_type == 'questionnaire':
                        # Questions is a list of questions, where each question contains an ID, text, and type
                        #   Type indicates what type of question it is; e.g. Instructional, Multiple choice, etc
                        questions = self.tasks[task]['questions']
                        for i in range(len(questions)):
                            # Question should be a dict
                            question = questions[i]
                            if type(question) == dict:
                                if 'type' in question.keys():
                                    question_format = question['type']['format']
                                elif 'default_type' in self.tasks[task].keys():
                                    question_format = self.tasks[task]['default_type']['format']
                                else:
                                    question_format = np.nan
                            else:
                                question = {'text':question}
                                question_format = np.nan

                            if question_format != 'instruction':
                                if "id" in question.keys():
                                    question_id = question['id']
                                else:
                                    question_id = "{}_{:0>2d}".format(task, i + 1)
                                if question_format in self.constants['NUMERICAL_QUESTION_TYPES']:
                                    numeric_q_cols.append(question_id)
                                if question_format in self.constants['CATEGORICAL_QUESTION_TYPES']:
                                    cat_q_cols.append(question_id)
                                q_cols.append(question_id)
                            else:
                                pass
                    elif task_type == 'aat':

                        num_practice = sum([len([i]) for i in self.stimulus_sets[self.tasks[task]['practice_targets'][0]]])
                        num_practice += sum([len([i]) for i in self.stimulus_sets[self.tasks[task]['practice_controls'][0]]])
                        
                        # Check if there is a specified number of repititions, if not default to 1. 
                        # target_rep = max(1, (self.tasks[task]['target_rep'] if isinstance(self.tasks[task]['target_rep'], (int, float)) else 0))
                        target_rep = max(1, float(self.tasks[task]['target_rep']))
                        num_stim = sum([len([i]) for i in self.stimulus_sets[self.tasks[task]['targets'][0]]]) * target_rep
                        # Check if there is a specified number of repititions, if not default to 1. 
                        # control_rep = max(1, (self.tasks[task]['control_rep'] if isinstance(self.tasks[task]['control_rep'], (int, float)) else 0))
                        control_rep = max(1, float(self.tasks[task]['control_rep']))
                        num_stim += sum([len([i]) for i in self.stimulus_sets[self.tasks[task]['controls'][0]]]) * control_rep

                        num_blocks = self.tasks[task]['amount_of_blocks']
                        cumulative_trials = 1

                        for b in range(num_blocks):
                            # Practice blocks are given by b = 0 and b = num_stim/2.
                            if b == 0 or b == num_blocks/2:
                                num_trials = int(num_practice/2.)
                                is_practice = True
                            else:
                                num_trials = int(num_stim/(num_blocks - 2))
                                is_practice = False

                            for t in range(0, num_trials):
                                sessionlst.append(session)
                                blocks.append(b + 1)
                                trials.append(t + 1)
                                total_trials.append(cumulative_trials)
                                practice.append(is_practice)
                                cumulative_trials += 1
                    else:
                        sessionlst.append(session)
                        blocks.append(np.nan)
                        trials.append(np.nan)
                        total_trials.append(np.nan)
                        practice.append('NA')

            cols = self.cols + sorted(q_cols) + self.aat_cols

            condition_table = pd.DataFrame(columns = cols, index = list(range(len(sessionlst))))

            # Fill in necessary columns
            cols_with_data = ['CONDITION_COLUMN', 'SESSION_COLUMN', 'BLOCK_COLUMN',
                              'IS_PRACTICE_COLUMN', 'TRIAL_NUMBER_COLUMN', 'TRIAL_NUMBER_CUM_COLUMN']
            # NOTE: Order in the data list should mirror that of the "cols_with_data" variable above
            data_list = [condition, sessionlst, blocks, practice, trials, total_trials]

            # for col in condition_table.columns:
            #     condition_table[col] = condition_table[col].apply(lambda x: [])

            for i in range(len(cols_with_data)):
                col = cols_with_data[i]
                d = data_list[i]
                entry = self.constants[col]
                condition_table[entry] = d

            # Set custom index for the dataframe for easy referencing when inputting AAT data. 
            # Structure as index = [Session Column, Block Column, Trial Number]
            # Example index: ['push_happy', '2', '10'] -> Push Happy session, block 2, trial 10. 
            condition_table = condition_table.set_index([self.constants['SESSION_COLUMN'], self.constants['BLOCK_COLUMN'], self.constants['TRIAL_NUMBER_COLUMN']]).sort_index()

            self.CondTablesDict.update({'{}'.format(condition) : condition_table})
            self.CategoricalQuestionColsDict.update({'{}'.format(condition) : cat_q_cols})


    # General function to load Json files.
    #   Inputs: path = Full file path (incl .json)
    #   Returns: json file data
    def LoadJson(self, path):
        if os.path.isfile(path):
            with open(path, 'r', encoding = 'utf-8') as f:
                self.jsonfile = json.loads(f.read(), strict = False)
            f.close()
        return self.jsonfile