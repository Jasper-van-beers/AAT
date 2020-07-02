#========================= IMPORTS =========================#
# External libraries, for getting directories
import os

# Import AAT module; if AAT is located in another folder, you
# may need to add library to path <sys.path.insert(0, AAT_Dir)>
# or navigate to the AAT folder, and import the AAT.py file
import AAT as AAT

#========================= DATA PROCESSING =========================#
# Get current working directory
cwd = os.getcwd()
# Get path to the data folder
DataFolders = cwd.split('\\')[:-4]
DataFoldersPath = '\\'.join(DataFolders)

# First we need to initialize the AAT DataImporter object
# During initialization, we need to provide the path to the folder 
# containing the experimental set-up data (i.e. conditions.json, 
# stimulus_sets.json, tasks.json, and sessions.json)
# Assuming folder structure of data from: https://osf.io/y5b32/
ExperimentalConditionsPath = '{}\\data\\external\\Study 2'.format(DataFoldersPath)
# We also need to provide the path to the folder containing the 
# raw AAT data. 
ParticipantDataPath = '{}\\data\\raw\\Study 2\\mobile'.format(DataFoldersPath)

# Initialize DataImporter. 
DataImporter = AAT.DataImporter(ExperimentalConditionsPath, ParticipantDataPath)
# After initializing the DataImporter, we can set various parameters
# by calling DataImporter.<parameter> = value, for the possible 
# parameters, see __init__ under the DataImporter class. 

# Import the raw data using ImportData()
RawData = DataImporter.ImportData()

# Since we are dealing with a lot of data, we will first try to remove
# missing data (NaNs) to reduce the scope, before carrying on with 
# filtering. Here, we use the RawData as the input. 
NoNaNData = DataImporter.FilterNaNs(RawData)

# We can now filter the data. We could also compute the reaction times
# first, but the FilterData() function will compute them automatically
# if they have not yet been calculated. 
# Here we can set various parameters regarding that to filter. In this
# case, we opt to remove 'unrealistic' accelerations as well. 
# FilterData(), by default, will also estimate the bias of the 
# accelerometer and attempt to correct for this
# FilterData() returns both the filtered dataframe, and a dataframe with
# the removed data. 
FilteredData, RemovedData = DataImporter.FilterData(NoNaNData, RemoveAcc = True)

# After filtering, we can resample the data to get a consistent time step, 
# and to align the gyroscope and accelerometer data with a common timeline. 
# This needs to be done BEFORE correcting the acceleration for rotations.
ResampledData = DataImporter.ResampleData(FilteredData)

# We can now apply rotation corrections to the accelerometer data
CorrectedData = DataImporter.Correct4Rotations(ResampledData)

# With the corrected accelerometer data, we can more reliabably compute
# the displacements of the phone. 
PreprocessedData = DataImporter.ComputeDistance(CorrectedData)

# NOTE: The functions above can also be called within each other
# For example:
# PreprocessedData = DH.ComputeDistance(DH.Correct4Rotations(DH.ResampleData(DH.FilterData(DH.ComputeRT(DH.FilterNaNs(DH.ImportData())), RemoveAcc=True)[0])))


# Since it takes quite a while to run this preprocessing, we can save 
# the dataframe to a file using SaveDF() from the DataImporter class.
SaveDirectory = '\\'.join(os.getcwd().split('\\')[:-1])
# print('[INFO] Saving PreprocessedData.pkl...')
# DataImporter.SaveDF('PreprocessedData.pkl', PreprocessedData, SaveDirectory)
# print('[INFO] Done.')
# When can then load the file using LoadDF() from the DataImporter
# class. See code below 

# PreprocessedData = DataImporter.LoadDF('PreprocessedData.pkl', SaveDirectory)

# Now that we have preprocessed the data, we can plot some results! 
TrialPoint = 5

# First, we can initialize the Plotter() class. To remain consistent
# with the DataImporter class, we can specify the constants used in 
# the DataImporter class. 
Plot = AAT.Plotter(constants = DataImporter.constants)

# For each plot, we need to supply the trial point, and the dataframe
# from which we will extract data. 
Plot.AccelerationTime(TrialPoint, PreprocessedData)

Plot.DistanceTime(TrialPoint, PreprocessedData)

# To show these plots, we call the ShowPlots() function
Plot.ShowPlots()

# We can also plot the data in 3D. Here, the starting point
# and ending point of the motion are shown. 
Plot.Acceleration3D(TrialPoint, PreprocessedData)

Plot.Trajectory3D(TrialPoint, PreprocessedData)

Plot.ShowPlots()

# Since it may be difficult to interpret the 3D plots due to the 
# lack of information in the time domain, we can also animate 
# these plots to see the motion over time. 
# In this case, we do not need to call the ShowPlots() function
Plot.AnimateAcceleration3D(TrialPoint, PreprocessedData)

Plot.AnimateTrajectory3D(TrialPoint, PreprocessedData)


# So far, we have only looked at individual trials of participants
# It is more insightful to look at their averages over the 
# various conditions. For example, the average of their Reaction
# Time for all Pull happy trials. To facilitate this,, we may use
# the Analysis class

# Initialize Analysis class. Since this class relies explicitly on
# the stimulus names, we need to specify the 'control' and 'target'
# stimuli names. In this case, they are 'angry' and 'happy' 
# respectively. 
# Again we can specify the DataImporter constants for consistency
Analysis = AAT.Analysis('angry', 'happy', constants = DataImporter.constants)

# To average the data (within participant) for each condition,
# we input the preprocessed data. We also need to specify the 
# controls (e.g. angry) and targets (e.g. happy) such that the
# relevant conditional columns can be generated. These should
# correspond to the control and target names in the raw data
# (note, that this is also case sensitive!)

WithinPAvgData = Analysis.AverageWithinParticipant(PreprocessedData)
# Now that the columns are different, we can inspect them. Here:
#       'PID' = Participant ID
#       'time' = Unified time array
#       'Acc ___ ___' = Acceleration for push/pull for control/target
#       'Dist ___ ___' = Distance for push/pull for control/target
#       'RT ___ ___' = Reaction time for push/pull for control/target
#       'DeltaA ___ ___' = Peak reaction acceleration
#       'DeltaD ___ ___' = Peak reaction distance
print('New column names: {}'.format(WithinPAvgData.columns))

# We can make use of the same plotting functions to illustrate this averaged data
# In this case, we can specify some additional parameters:
#       axis = ['X', 'Y', 'Z'] -> What axes to plot
#       movement = ['Push', 'Pull'] -> What movements to plot
#       stimulus = ['control', 'target'] -> What stimuli to plot 
ParticipantNum = 20 # Example participant
Plot.AccelerationTime(ParticipantNum, WithinPAvgData, axis = ['Z'], movement = ['Pull'], stimulus = ['happy', 'angry'])
Plot.DistanceTime(ParticipantNum, WithinPAvgData, axis = ['Z'], movement = ['Pull', 'Push'], stimulus = ['angry'])
Plot.ShowPlots()

# The 3D plots follow a similar format, but without specifying the axis
Plot.Acceleration3D(ParticipantNum, WithinPAvgData, movement = ['Pull'], stimulus = ['happy', 'angry'])
Plot.Trajectory3D(ParticipantNum, WithinPAvgData, movement = ['Pull', 'Push'], stimulus = ['angry'])
Plot.ShowPlots()

# Likewise, we can animate these movements
Plot.AnimateAcceleration3D(ParticipantNum, WithinPAvgData, UseBlit=True, movement = ['Pull'], stimulus=['happy', 'angry'])
Plot.AnimateTrajectory3D(ParticipantNum, WithinPAvgData, UseBlit=True, movement = ['Pull'], stimulus=['happy', 'angry'])

# We can go one step further, and compute the averages across pariticpants
# To do this, we make use of the following function. Here, we must input the 
# data that has already been averaged within participants 
AveragedData = Analysis.AverageBetweenParticipant(WithinPAvgData)

# In this case, there is only one 'participant' (PID = 'Average') therefore:
ParticipantNum = 0
# Again we may plot the same plots using the Plotter class
Plot.AccelerationTime(ParticipantNum, AveragedData, axis = ['Z'], movement = ['Pull'], stimulus = ['happy', 'angry'])
Plot.DistanceTime(ParticipantNum, AveragedData, axis = ['Z'], movement = ['Pull', 'Push'], stimulus = ['angry'])

Plot.Acceleration3D(ParticipantNum, AveragedData, movement = ['Pull'], stimulus = ['happy', 'angry'])
Plot.Trajectory3D(ParticipantNum, AveragedData, movement = ['Pull', 'Push'], stimulus = ['angry'])
Plot.ShowPlots()

Plot.AnimateAcceleration3D(ParticipantNum, AveragedData, UseBlit=True, movement = ['Pull'], stimulus=['happy', 'angry'])
Plot.AnimateTrajectory3D(ParticipantNum, AveragedData, UseBlit=True, movement = ['Pull'], stimulus=['happy', 'angry'])



# Another plotting utility that we have yet to discuss is the ApproachAvoidanceXZ plot
# In this case, plot the trajectory/acceleration of the movement in the X and Z axis
# (Think of it as looking at the participant from the side, where x is the vertical 
# movment and z is the lateral movement). This function works with the preprocessed data
# or the averaged data 
# Here, we need to add an additional parameter:
#       metric = 'distance' or 'acceleration' -> indicates which trace to plot
# If distance is selected, then the plot also shows the (approximate) regions in the graph
# which correspond to an overall 'avoidance' or 'approach' movement (Based on the distance
# between the user and the device)
Plot.ApproachAvoidanceXZ('distance', ParticipantNum, AveragedData, movement =  ['Pull', 'Push'], stimulus = ['happy', 'angry'])
Plot.ShowPlots()

# If we wish to plot multiple plots in one figure, this is also possible with the 
# MultiPlot function (Note: this does not work for the animations).
# In this case, we need input the following:
#       Layout = A tuple of the grid that we want to use: (nrows, ncolumns)
#       Functions = List of plot functions
#       FuncArgs = List of dictionaries containing the function arguments, 
#                   the order should correspond with the list of functions

# Example 2x2 plot
Funcs = [Plot.AccelerationTime, Plot.Acceleration3D, Plot.DistanceTime, Plot.Trajectory3D]
FuncArgs = [{'participant':ParticipantNum, 'DF':AveragedData, 'axis':['Z'], 'movement':['Pull'], 'stimulus':['happy', 'angry']},
            {'participant':ParticipantNum, 'DF':AveragedData, 'movement':['Pull'], 'stimulus':['happy', 'angry']},
            {'participant':ParticipantNum, 'DF':AveragedData, 'axis':['Z'], 'movement':['Pull'], 'stimulus':['happy', 'angry']},
            {'participant':ParticipantNum, 'DF':AveragedData, 'movement':['Pull'], 'stimulus':['happy', 'angry']}]

Plot.MultiPlot((2, 2), Funcs, FuncArgs)
Plot.ShowPlots()

# Example 2x1 plot
Funcs = [Plot.AccelerationTime, Plot.ApproachAvoidanceXZ]
FuncArgs = [{'participant':ParticipantNum, 'DF':AveragedData, 'axis':['Z'], 'movement':['Pull'], 'stimulus':['happy', 'angry']},
            {'metric':'distance','participant':ParticipantNum, 'DF':AveragedData, 'movement':['Pull'], 'stimulus':['happy', 'angry']}]

Plot.MultiPlot((2, 1), Funcs, FuncArgs)
Plot.ShowPlots()