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
# with the DataImporter class, we need to input the constants from 
# the DataImporter class
Plot = AAT.Plotter(DataImporter.constants)

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