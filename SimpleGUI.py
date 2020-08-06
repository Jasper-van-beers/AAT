import PySimpleGUI as sg
import os
import json
import numpy as np
import xlsxwriter as xls
import pandas as pd

import AAT as AAT
#=======================================================================================================================#
# https://pysimplegui.readthedocs.io/en/latest/cookbook/
#=======================================================================================================================#

def StartWindow(Title, config = None):

    def CheckStatus(event, values, defaults):
        disabled = {}
        for key in values.keys():
            disabled.update({key:False})
        if values[event] != defaults[event]:
            # If value is True then disable other checkbox options
            if values[event]:
                for key in values.keys():
                    if key != event:
                        disabled.update({key:True})

        return disabled, values

    isDefault = [True, False, False, False]
    isDisabled = [not i for i in isDefault]

    if config is not None:
        if int(config['Analysis Method']) in (0, 1, 2, 3):
            isDefault = [False, False, False, False]
            isDefault[int(config['Analysis Method'])] = True
            isDisabled = [not i for i in isDefault]

    layout = [[sg.Text('Please select analysis method', font = ('Helvetica', 12))],
              [sg.Text('')],
              [sg.Checkbox('Import raw data and run both processing and analysis', default = isDefault[0], disabled = isDisabled[0], key = '-BOTH_RAW-', enable_events=True)],
              [sg.Checkbox('Import raw data and run processing only', key = '-PRE_RAW-', default = isDefault[1], disabled = isDisabled[1], enable_events=True)],
              [sg.Checkbox('Import from file and run processing only', key = '-PRE_IMPORT-', default = isDefault[2], disabled = isDisabled[2], enable_events=True)],
              [sg.Checkbox('Import from file and run analysis only', key = '-ANA_IMPORT-', default = isDefault[3], disabled = isDisabled[3], enable_events=True)],
              [sg.Text('')],
              [sg.Text('')],
              [sg.Cancel(key='-CANCEL-'), sg.Button('Next', key='-NEXT-', disabled=False)]]

    window = sg.Window(Title, layout)

    defaults = {'-BOTH_RAW-':False, '-PRE_RAW-':True, '-PRE_IMPORT-':True, '-ANA_IMPORT-':True}
    mapping = {'-BOTH_RAW-':0, '-PRE_RAW-':1, '-PRE_IMPORT-':2, '-ANA_IMPORT-':3}
    cancelled = False
    while True:
        event, value = window.read()
        if event in (sg.WIN_CLOSED, '-CANCEL-'):
            cancelled = True
            break
        if event == '-NEXT-':
            break
        isDisabled, defaults = CheckStatus(event, value, defaults)
        for key in defaults.keys():
            window[key].update(disabled=isDisabled[key])
        if all(defaults[key] == False for key in defaults.keys()):
            window['-NEXT-'].update(disabled=True)
        else:
            window['-NEXT-'].update(disabled=False)

    if not cancelled:
        for key in value.keys():
            if value[key] and key in defaults.keys():
                RunAll = mapping[key]
                break
    else:
        RunAll = None

    window.close()

    return RunAll



def PreProcessFolder(Title, config = None):
    fieldEntrySize = (100, 1)

    inputFieldKeys = ['-RAW_DATA_DIR-', '-COND_DATA_DIR-']

    defaults = {'-RAW_DATA_DIR-':'',
                '-COND_DATA_DIR-':'',
                '-SAVE-':False,
                '-SAVE_FILENAME-':'My_Processed_AAT_Data',
                '-SAVE_DIR-':'', 
                '-SMALLSAVE-':False,
                '-SAVECSV-':False}
    isDisabled = True
    if config is not None:
        Dirs = config['Processing Directories']
        defaults.update({'-RAW_DATA_DIR-':Dirs['Raw Data Path'],
                         '-COND_DATA_DIR-':Dirs['Conditions File Path']})
            
        if Dirs['Save File']:
            defaults.update({'-SAVE_DIR-':Dirs['Save Path']})
            defaults.update({'-SAVE-':Dirs['Save File']})
            defaults.update({'-SAVE_FILENAME-':Dirs['Save Filename']})
            defaults.update({'-SMALLSAVE-':Dirs['Save Condensed']})
            defaults.update({'-SAVECSV-':Dirs['Save as .csv']})
            if all(os.path.isdir(defaults[key]) for key in inputFieldKeys) and os.path.isdir(Dirs['Save Path']):
                isDisabled = False
        else:
            if all(os.path.isdir(defaults[key]) for key in inputFieldKeys):
                isDisabled = False

    bColors = ['grey42', 'white']
    tColors = ['grey42', 'black']

    # Build window layout
    layout = [[sg.Text('Please locate the raw data folder: ')],
              [sg.In(default_text=defaults['-RAW_DATA_DIR-'], size=fieldEntrySize, key = '-RAW_DATA_DIR-', enable_events=True), sg.FolderBrowse()],
              [sg.Text('')],
              [sg.Text('Please locate the conditions folder: ')],
              [sg.In(default_text=defaults['-COND_DATA_DIR-'], size=fieldEntrySize, key = '-COND_DATA_DIR-', enable_events=True), sg.FolderBrowse()],
              [sg.Text('')],
              [sg.Checkbox('Would you like to save the processed files?', key='-SAVE-', enable_events=True, default=defaults['-SAVE-'])],
              [sg.InputText(default_text=defaults['-SAVE_DIR-'], size=fieldEntrySize, disabled=not defaults['-SAVE-'], key = '-SAVE_DIR-', enable_events=True, text_color=tColors[int(defaults['-SAVE-'])]), sg.FolderBrowse(disabled=not defaults['-SAVE-'], key = '-SAVE_BROWSE-')],
              [sg.Text('Save filename:', text_color=bColors[int(defaults['-SAVE-'])], key='-SF_TEXT-'), sg.InputText(default_text=defaults['-SAVE_FILENAME-'], size=(50, 1), key='-SAVE_FILENAME-', disabled = not defaults['-SAVE-'], text_color=tColors[int(defaults['-SAVE-'])])],
              [sg.Checkbox('Save condensed data', key='-SMALLSAVE-', enable_events = True, default=defaults['-SMALLSAVE-'], disabled=not defaults['-SAVE-']),
               sg.Checkbox('Save as .csv', key='-SAVECSV-', enable_events = True, default=defaults['-SAVECSV-']*defaults['-SMALLSAVE-'], disabled=not defaults['-SMALLSAVE-'])],
              [sg.Cancel(key = '-CANCEL-'), sg.Button('Next', key = '-NEXT-', disabled=isDisabled)]]

    window = sg.Window(Title, layout)

    cancelled = False
    while True:
        event, value = window.read()
        if event in (sg.WIN_CLOSED, '-CANCEL-'):
            cancelled = True
            break
        if event == '-NEXT-':
            break
        if event == '-SAVE-':
            if value[event]:
                window['-SAVE_DIR-'].update(disabled=False, text_color=tColors[int(value['-SAVE-'])])
                window['-SAVE_BROWSE-'].update(disabled=False)
                window['-SAVE_FILENAME-'].update(disabled=False, text_color=tColors[int(value['-SAVE-'])])
                window['-SF_TEXT-'].update(text_color = bColors[int(value['-SAVE-'])])
                window['-SMALLSAVE-'].update(disabled=False)
                window['-SAVECSV-'].update(disabled=not value['-SMALLSAVE-'])
            else:
                window['-SAVE_DIR-'].update(disabled=True, text_color=tColors[int(value['-SAVE-'])])
                window['-SAVE_BROWSE-'].update(disabled=True)
                window['-SAVE_FILENAME-'].update(disabled=True, text_color=tColors[int(value['-SAVE-'])])
                window['-SF_TEXT-'].update(text_color = bColors[int(value['-SAVE-'])])
                window['-SMALLSAVE-'].update(value=False, disabled=True)
                window['-SAVECSV-'].update(value=False, disabled=True)
        if event == '-SMALLSAVE-':
            if value[event]:
                window['-SAVECSV-'].update(disabled=False)
            else:
                window['-SAVECSV-'].update(value=value['-SMALLSAVE-'], disabled=True)
        if value['-SAVE-']:
            if all(os.path.isdir(value[key]) for key in inputFieldKeys) and os.path.isdir(value['-SAVE_DIR-']):
                window['-NEXT-'].update(disabled=False)
            else:
                window['-NEXT-'].update(disabled=True)
        else:
            if all(os.path.isdir(value[key]) for key in inputFieldKeys):
                window['-NEXT-'].update(disabled=False)
            else:
                window['-NEXT-'].update(disabled=True)
    
    window.close()

    # If save is deselected, clear all saveparams
    if not value['-SAVE-']:
        value['-SAVE_DIR-'] = None
        value['-SAVE_FILENAME-'] = None
        value['-SMALLSAVE-'] = None
        value['-SAVECSV-'] = None

    if cancelled:
        value = None

    return value



def LoadFromFile(Title, config = None):
    fieldEntrySize = (100, 1)

    defaults = {'-FILE_PATH-':'',
                '-SAVE-':False,
                '-SAVE_DIR-':'', 
                '-SAVE_FILENAME-':'Processed_AAT',
                '-SMALLSAVE-':False,
                '-SAVECSV-':False}
    isDisabled = True
    if config is not None:
        Dirs = config['Import File Information']
        defaults.update({'-FILE_PATH-':Dirs['File Path']})
            
        if Dirs['Save File']:
            defaults.update({'-SAVE_DIR-':Dirs['Save Path']})
            defaults.update({'-SAVE-':Dirs['Save File']})
            defaults.update({'-SAVE_FILENAME-':Dirs['Save Filename']})
            defaults.update({'-SMALLSAVE-':Dirs['Save Condensed']})
            defaults.update({'-SAVECSV-':Dirs['Save as .csv']})
            if os.path.isfile(Dirs['File Path']) and os.path.isdir(Dirs['Save Path']):
                isDisabled = False
        
        elif os.path.isfile(Dirs['File Path']):
            isDisabled = False

    bColors = ['grey42', 'white']
    tColors = ['grey42', 'black']

    # Build window layout
    layout = [[sg.Text('Please locate the File Path: ')],
              [sg.In(default_text = defaults['-FILE_PATH-'], size=fieldEntrySize, key = '-FILE_PATH-', enable_events=True), sg.FileBrowse(file_types=(('Pickle Files', '*.pkl'),))],
              [sg.Checkbox('Would you like to save the output files?', default = defaults['-SAVE-'], key='-SAVE-', enable_events=True)],
              [sg.InputText(default_text=defaults['-SAVE_DIR-'], size=fieldEntrySize, disabled=not defaults['-SAVE-'], key = '-SAVE_DIR-', enable_events=True, text_color=tColors[int(defaults['-SAVE-'])]), sg.FolderBrowse(disabled=not defaults['-SAVE-'], key = '-SAVE_BROWSE-')],
              [sg.Text('Save filename:', text_color=bColors[int(defaults['-SAVE-'])], key='-SF_TEXT-'), sg.InputText(default_text=defaults['-SAVE_FILENAME-'], size=(50, 1), key='-SAVE_FILENAME-', disabled = not defaults['-SAVE-'], text_color=tColors[int(defaults['-SAVE-'])])],
              [sg.Checkbox('Save condensed data', key='-SMALLSAVE-', enable_events = True, default=defaults['-SMALLSAVE-'], disabled=not defaults['-SMALLSAVE-']),
               sg.Checkbox('Save as .csv', key='-SAVECSV-', enable_events = True, default=defaults['-SAVECSV-']*defaults['-SMALLSAVE-'], disabled=not defaults['-SMALLSAVE-'])],              
              [sg.Cancel(key = '-CANCEL-'), sg.Button('Next', key = '-NEXT-', disabled=isDisabled)]]

    window = sg.Window(Title, layout)

    cancelled = False
    while True:
        event, value = window.read()
        if event in (sg.WIN_CLOSED, '-CANCEL-'):
            cancelled = True
            break
        if event == '-NEXT-':
            break
        if event == '-SAVE-':
            if value[event]:
                window['-SAVE_DIR-'].update(disabled=False, text_color=tColors[int(value['-SAVE-'])])
                window['-SAVE_BROWSE-'].update(disabled=False)
                window['-SAVE_FILENAME-'].update(disabled=False, text_color=tColors[int(value['-SAVE-'])])
                window['-SF_TEXT-'].update(text_color = bColors[int(value['-SAVE-'])])
                window['-SMALLSAVE-'].update(disabled=False)
                window['-SAVECSV-'].update(disabled=not value['-SMALLSAVE-'])
            else:
                window['-SAVE_DIR-'].update(disabled=True, text_color=tColors[int(value['-SAVE-'])])
                window['-SAVE_BROWSE-'].update(disabled=True)
                window['-SAVE_FILENAME-'].update(disabled=True, text_color=tColors[int(value['-SAVE-'])])
                window['-SF_TEXT-'].update(text_color = bColors[int(value['-SAVE-'])])
                window['-SMALLSAVE-'].update(value = False, disabled=True)
                window['-SAVECSV-'].update(value = False, disabled=True)
        if event == '-SMALLSAVE-':
            if value[event]:
                window['-SAVECSV-'].update(disabled=False)
            else:
                window['-SAVECSV-'].update(value=value['-SMALLSAVE-'], disabled=True)
        if value['-SAVE-']:
            if os.path.isfile(value['-FILE_PATH-']) and os.path.isdir(value['-SAVE_DIR-']):
                window['-NEXT-'].update(disabled = False)
            else:
                window['-NEXT-'].update(disabled = True)
        else:
            if os.path.isfile(value['-FILE_PATH-']):
                window['-NEXT-'].update(disabled = False)
            else:
                window['-NEXT-'].update(disabled = True)            

    window.close()

    # If save is deselected, clear all saveparams
    if not value['-SAVE-']:
        value['-SAVE_DIR-'] = None
        value['-SAVE_FILENAME-'] = None
        value['-SMALLSAVE-'] = None
        value['-SAVECSV-'] = None

    if cancelled:
        value = None

    return value



def LoadData(Title, FilePath):

    layout = [
        [sg.Text('Loading data from file', font=('Helvetica', 12))],
        [sg.Text('This may take a while, and the window may appear to stop responding. This is normal, please be patient.')],
        [sg.Text('This window will close automatically once the file is loaded.')],
        [sg.Cancel(key='-CANCEL-')]
    ]

    window = sg.Window(Title, layout)

    cancelled = False
    while True:
        event, value = window.read(timeout = 1)
        if event in (sg.WIN_CLOSED, '-CANCEL-'):
            cancelled = True
            break
        Data = pd.read_pickle(FilePath)
        break

    window.close()

    if cancelled:
        Data = None    

    return Data



def Pre_FunctionParams(Title, config = None, ImportFilePath = None, ForAnalysis = False):

    def getFunctionOptionalParams(f):
        numInputArgs = f.__code__.co_argcount
        AllInputs = f.__code__.co_varnames[1:numInputArgs]
        fDefaults = f.__defaults__
        fArgs = AllInputs[-len(fDefaults):]
        OptionalParams = {k:fDefaults[v] for v,k in enumerate(fArgs)}
        return OptionalParams


    def MapAxes(XYZ):
        Ref = ['X', 'Y', 'Z']
        Map = []
        for idx, active in enumerate(XYZ):
            if active:
                Map.append(Ref[idx])
        return Map


    def MapDefaultRun(defaultrun):
        order = [None] * int(len(defaultrun.keys()) + 1)
        Mapping = {'FILT':1, 'RT':0, 'ROTCORRECT':3, 'DISTANCE':4, 'RFRD':5}
        for func in defaultrun.keys():
            if defaultrun[func]:
                order[Mapping[func]] = func
        
        if 'FILT' in order:
            order[2] = 'RESAMPLE'

        sortedOrder = [func for func in order if func]

        return sortedOrder

    
    def MapExisitingOrder(OrderFromFile):
        Dependencies = {'FILT':'RT', 'ROTCORRECT':'FILT', 'DISTANCE':'FILT', 'RFRD':'FILT', 'RESAMPLE':'FILT'}
        Disabled = {'FILT':False, 'RT':False, 'ROTCORRECT':False, 'DISTANCE':False, 'RFRD':False}
        isRun = {'FILT':True, 'RT':True, 'ROTCORRECT':True, 'DISTANCE':True, 'RFRD':True}
        for tag in OrderFromFile:
            if tag in isRun.keys():
                Disabled[tag] = True
                isRun[tag] = False
                if tag in Dependencies.keys():
                    Disabled[Dependencies[tag]] = True
                    isRun[Dependencies[tag]] = False
        return Disabled, isRun


    def CheckOrder(order, which, exceptions = []):
        Dependencies = {'RT':None, 'FILT':'RT', 'ROTCORRECT':'FILT', 'DISTANCE':'FILT', 'RFRD':'FILT', 'RESAMPLE':'FILT'}
        FunctionMap = {'FILT':'Run filtering', 'RT':'Compute reaction times', 'ROTCORRECT':'Correct for rotations', 
                       'DISTANCE':'Compute distances', 'RFRD':'Compute reaction force and distance'}
        if which[1]:
            Dependencies.update({'RFRD':'DISTANCE'})

        for idx, func in enumerate(order):
            if Dependencies[func] is not None:
                if Dependencies[func] not in order[0:(idx + 1)]:
                    if Dependencies[func] not in exceptions:
                        order = None
                        sg.PopupError(f'ERROR\nThe selected function <{FunctionMap[func]}> depends on function <{FunctionMap[Dependencies[func]]}>')
                        break

        return order
    
    # Default processing parameters
    DefaultFilteringParams = getFunctionOptionalParams(AAT.DataImporter.FilterData)
    DefaultComputeRTParams = getFunctionOptionalParams(AAT.DataImporter.ComputeRT)
    DefaultRotCorrectParams = getFunctionOptionalParams(AAT.DataImporter.Correct4Rotations)
    DefaultDistCompParams = getFunctionOptionalParams(AAT.DataImporter.ComputeDistance)
    DefaultRFandRDParams = getFunctionOptionalParams(AAT.DataImporter.ComputeDeltaAandD)

    DefaultRun = {'FILT':True, 'RT':True, 'ROTCORRECT':True, 'DISTANCE':True, 'RFRD':True}

    # Replace defaults with config defaults, if it a config file is specified
    if config is not None:
        # Processing parameters config
        configProcessing = config['Default Processing Parameters']
        DefaultFilteringParams.update({'MinDataRatio':float(configProcessing['Filtering Parameters']['Minimum ratio of valid trials']),
                                        'RTThreshold':float(configProcessing['Filtering Parameters']['Reaction time cutoff']),
                                        'RemoveRT':configProcessing['Filtering Parameters']['Filter by reaction time'],
                                        'RemoveAcc':configProcessing['Filtering Parameters']['Filter by acceleration'],
                                        'CalibrateAcc':configProcessing['Filtering Parameters']['Correct for acceleration offset'],
                                        'RemoveIncorrectResponses':configProcessing['Filtering Parameters']['Remove incorrect responses']
                                        })

        DefaultComputeRTParams.update({'RTResLB':float(configProcessing['Reaction Time Parameters']['Minimum acceleration for a reaction']),
                                       'RTPeakDistance':float(configProcessing['Reaction Time Parameters']['(Assumed) Peak height width'])
                                        })

        DefaultRotCorrectParams.update({'StoreTheta':configProcessing['Rotation Correction Parameters']['Store Rotation Angles']})
        
        DefaultDistCompParams.update({'ComputeTotalDistance':configProcessing['Distance Computation Parameters']['Compute total distance magnitude']})
        
        _axes = [configProcessing['Reaction Force and Distance Parameters']['X (Vertical Axis)'], 
                 configProcessing['Reaction Force and Distance Parameters']['Y (Lateral Axis)'],
                 configProcessing['Reaction Force and Distance Parameters']['Z (Ventral Axis)']]
        _axes = MapAxes(_axes)
        _which = [configProcessing['Reaction Force and Distance Parameters']['Compute reaction forces'],
                  configProcessing['Reaction Force and Distance Parameters']['Compute reaction distances']]
        which = []
        if _which[0]:
            which.append('acceleration')
        if _which[1]:
            which.append('distance')
        DefaultRFandRDParams.update({'axes':_axes,
                                     'absolute':configProcessing['Reaction Force and Distance Parameters']['Use absolute values'],
                                     'which':which})

        DefaultRun = {'FILT':configProcessing['Filtering Parameters']['Run'],
                      'RT':configProcessing['Reaction Time Parameters']['Run'],
                      'ROTCORRECT':configProcessing['Rotation Correction Parameters']['Run'],
                      'DISTANCE':configProcessing['Distance Computation Parameters']['Run'],
                      'RFRD':configProcessing['Reaction Force and Distance Parameters']['Run']}

    ForceFunctionSelection = False

    oldOrder = []

    if ImportFilePath is not None:
        MetaDataFilePath = ImportFilePath[:-4] + '_metadata.json'
        ForceFunctionSelection = True
        if os.path.isfile(MetaDataFilePath):
            with open(MetaDataFilePath, 'r') as f:
                ImportParams = json.loads(f.read())
            PermaDisabled, DefaultRun = MapExisitingOrder(ImportParams['TagList'])
            oldOrder = ImportParams['TagList']
        else:

            PermaDisabled = {'FILT':False, 'RT':False, 'ROTCORRECT':False, 'DISTANCE':False, 'RFRD':False}
            DefaultRun = {'FILT':True, 'RT':True, 'ROTCORRECT':True, 'DISTANCE':True, 'RFRD':True}

            sg.PopupError('ERROR\n Metadata for the imported file \n\t<{}>\n could not be found.\n Some errors may arise during preprocessing due to this.'.format(ImportFilePath))

            ErLayout = [
                [sg.Text('Would you like me to infer the metadata, or run without this information?', font = ('Helvetica', 12), key = '-MESSAGE-')],
                [sg.Text('Note: When running without metadata information, some errors may arise during preprocessing', key = '-NOTE-')],
                [sg.Text('', key='-NOTE2-')],
                [sg.Button('Infer metadata', key='-INFER-'), sg.Button('Continue without metadata', key = '-BLIND-')]
            ]

            win = sg.Window(Title, ErLayout)

            cancelled = False
            while True:
                event, value = win.read()
                if event in (sg.WIN_CLOSED, '-CANCEL-'):
                    cancelled = True
                    break
                if event == '-INFER-':
                    DataImporter = AAT.DataImporter(None, None, printINFO=False)
                    win['-MESSAGE-'].update(value = 'Loading file...')
                    win['-NOTE-'].update(value = 'This may take a while, and the window may (briefly) stop responding.')
                    win['-NOTE2-'].update(value = 'This is normal, please wait...')
                    win.refresh()
                    _dummy = DataImporter.LoadDF(ImportFilePath, None)
                    if DataImporter.constants['RT_COLUMN'] in _dummy.columns:
                        PermaDisabled.update({'RT':True})
                        DefaultRun.update({'RT':False})
                    if DataImporter.constants['DISTANCE_Z_COLUMN'] in _dummy.columns:
                        PermaDisabled.update({'DISTANCE':True})
                        DefaultRun.update({'DISTANCE':False})
                    if DataImporter.constants['DELTA_A_COLUMN'] in _dummy.columns or DataImporter.constants['DELTA_D_COLUMN'] in _dummy.columns:
                        PermaDisabled.update({'RFRD':True})
                        DefaultRun.update({'RFRD':False})
                    
                    resample = True
                    for t in range(len(_dummy['times'][0]) - 1):
                        dt = _dummy['times'][0][t + 1] - _dummy['times'][0][t]
                        if dt > 1:
                            resample = False
                            break
                    if resample:
                        PermaDisabled.update({'FILT':True})
                        DefaultRun.update({'FILT':False})
                    break

                if event == '-BLIND-':
                    break
            win.close()
    else:
        PermaDisabled = {'FILT':False, 'RT':False, 'ROTCORRECT':False, 'DISTANCE':False, 'RFRD':False}


    isDisabled = {i : (not DefaultRun[i]) for i in DefaultRun.keys()}

    # Window layout
    tColors = ['grey60', 'black']
    bColors = ['grey42', 'white']

    # PROCESSING PARAMETERS LAYOUT
    FilteringLayout = [[sg.Column([[sg.Text('Minimum ratio of valid trials', key='T_MinDataRatio', text_color=bColors[int(DefaultRun['FILT'])])],
                                    [sg.Text('Reaction time cutoff', key='T_RTThreshold', text_color=bColors[int(DefaultRun['FILT'])])]]),
                        sg.Column([[sg.InputText(default_text='{}'.format(DefaultFilteringParams['MinDataRatio']), key='MinDataRatio', disabled=isDisabled['FILT'], use_readonly_for_disable=True, text_color=tColors[int(DefaultRun['FILT'])]), sg.Text('[-]', key = 'U_MinDataRatio', text_color=bColors[int(DefaultRun['FILT'])])],
                                    [sg.InputText(default_text='{}'.format(DefaultFilteringParams['RTThreshold']), key='RTThreshold', disabled=isDisabled['FILT'], use_readonly_for_disable=True, text_color=tColors[int(DefaultRun['FILT'])]), sg.Text('[ms]', key = 'U_RTThreshold', text_color=bColors[int(DefaultRun['FILT'])])]]),
                        sg.Column([[sg.Checkbox('Filter by reaction time', default = DefaultFilteringParams['RemoveRT'], key='RemoveRT', disabled=isDisabled['FILT'])], 
                                   [sg.Checkbox('Filter by acceleration', default = DefaultFilteringParams['RemoveAcc'], key='RemoveAcc', disabled=isDisabled['FILT'])],
                                   [sg.Checkbox('Correct for acceleration offset', default = DefaultFilteringParams['CalibrateAcc'], key='CalibrateAcc', disabled=isDisabled['FILT'])],
                                   [sg.Checkbox('Remove incorrect responses', default = DefaultFilteringParams['RemoveIncorrectResponses'], key='RemoveIncorrectResponses', disabled=isDisabled['FILT'])]])]]

    ReactionTimeLayout = [[sg.Column([[sg.Text('Minimum acceleration for a reaction', key = 'T_RTResLB', text_color=bColors[int(DefaultRun['RT'])])],
                                     [sg.Text('(Assumed) Peak height width', key = 'T_RTPeakDistance', text_color=bColors[int(DefaultRun['RT'])])]]),
                           sg.Column([[sg.InputText(default_text='{}'.format(DefaultComputeRTParams['RTResLB']), key='RTResLB', disabled=isDisabled['RT'], use_readonly_for_disable=True, text_color=tColors[int(DefaultRun['RT'])]), sg.Text('[m/s^-2]', key = 'U_RTResLB', text_color=bColors[int(DefaultRun['RT'])])],
                                      [sg.InputText(default_text='{}'.format(DefaultComputeRTParams['RTPeakDistance']), key='RTPeakDistance', disabled=isDisabled['RT'], use_readonly_for_disable=True, text_color=tColors[int(DefaultRun['RT'])]), sg.Text('[ms]', key = 'U_RTPeakDistance', text_color=bColors[int(DefaultRun['RT'])])]])]]

    RotCorrectLayout = [[sg.Column([[sg.Checkbox('Store Rotation Angles', default=DefaultRotCorrectParams['StoreTheta'], key='StoreTheta', disabled=isDisabled['ROTCORRECT'])]])]]

    DistCompLayout = [[sg.Column([[sg.Checkbox('Compute total distance magnitude', default=DefaultDistCompParams['ComputeTotalDistance'], key='ComputeTotalDistance', disabled=isDisabled['DISTANCE'])]])]]

    RFandRDLayout = [[sg.Frame('Axes', [[sg.Column([[sg.Checkbox('X (Vertical Axis)', default = 'X' in DefaultRFandRDParams['axes'], key='-X-', disabled=isDisabled['RFRD']),
                                  sg.Checkbox('Y (Lateral Axis)', default = 'Y' in DefaultRFandRDParams['axes'], key='-Y-', disabled=isDisabled['RFRD']),
                                  sg.Checkbox('Z (Ventral Axis)', default = 'Z' in DefaultRFandRDParams['axes'], key='-Z-', disabled=isDisabled['RFRD'])]])]])],
                      [sg.Column([[sg.Checkbox('Use absolute values', default = DefaultRFandRDParams['absolute'], key='absolute', disabled=isDisabled['RFRD'])]]),
                       sg.Column([[sg.Checkbox('Compute Reaction Force', default = 'acceleration' in DefaultRFandRDParams['which'], key='-RF-', disabled=isDisabled['RFRD'])],
                                 [sg.Checkbox('Compute Reaction Distance', default = 'distance' in DefaultRFandRDParams['which'], key='-RD-', disabled=isDisabled['RFRD'])]])]]


    Paramlayout = [
              [sg.Text('')],
              [sg.Checkbox('Compute reaction times', default=DefaultRun['RT'], key='-RUN1-', enable_events=True, disabled=PermaDisabled['RT']), 
               sg.Checkbox('Run filtering', default=DefaultRun['FILT'], key='-RUN0-', enable_events = True, disabled=PermaDisabled['FILT']),
               sg.Checkbox('Correct for rotations', default=DefaultRun['ROTCORRECT'], key='-RUN2-', enable_events = True, disabled=PermaDisabled['ROTCORRECT']),
               sg.Checkbox('Compute distances', default=DefaultRun['DISTANCE'], key = '-RUN3-', enable_events=True, disabled=PermaDisabled['DISTANCE']),
               sg.Checkbox('Compute reaction force and distance', default=DefaultRun['RFRD'], key= '-RUN4-', enable_events=True, disabled=PermaDisabled['RFRD'])],
              [sg.Frame(' Reaction Time Parameters ', ReactionTimeLayout, relief=sg.RELIEF_SUNKEN, key = '-RT_PARAMS-')],
              [sg.Frame(' Filtering Parameters ', FilteringLayout, relief=sg.RELIEF_SUNKEN, key = '-FILTERING_PARAMS-')],
              [sg.Frame(' Reaction Force and Distance Parameters ', RFandRDLayout, relief=sg.RELIEF_SUNKEN, key = '-RFRD_PARAMS-'),
               sg.Column([[sg.Frame(' Rotation Correction Parameters ', RotCorrectLayout, relief=sg.RELIEF_SUNKEN, key = '-ROT_PARAMS-')],
                          [sg.Text('')],
                          [sg.Frame(' Distance Computation Parameters ', DistCompLayout, relief=sg.RELIEF_SUNKEN, key = '-DIST_PARAMS-')]])]
             ]


    layout = [
        [sg.TabGroup([[sg.Tab('Processing Parameters', Paramlayout, font = ('Helvetica', 12), tooltip='Please specify the processing parameters (e.g. filtering) here')]])],
        [sg.Cancel(key = '-CANCEL-'), sg.Button('Next', key = '-NEXT-')]
        ]

    window = sg.Window(Title, layout)

    cancelled = False
    while True:
        event, value = window.read()
        if event in (sg.WIN_CLOSED, '-CANCEL-'):
            cancelled = True
            break
        if event == '-NEXT-':
            DefaultRun = {i : not isDisabled[i] for i in isDisabled.keys()}
            order = MapDefaultRun(DefaultRun)
            order = CheckOrder(order, [value['-RF-'], value['-RD-']], exceptions = MapDefaultRun(PermaDisabled))
            if order is not None:
                if ForAnalysis and 'FILT' not in order:
                    sg.PopupError('ERROR\n User has selected to run analysis. The analysis cannot be run unless <Run Filtering> is selected here.')
                else:
                    if DefaultRun['RFRD']:
                        # Check that user has indicated to compute RF and/or RD
                        if value['-RF-'] or value['-RD-']:
                            value.update({'order':order})
                            # Check that at least 1-axis has been selected:
                            if all(not i for i in [value['-X-'], value['-Y-'], value['-Z-']]):
                                sg.PopupError('ERROR\nUser selected to <Compute reaction force and distance>\nHowever, no axis has been chosen for which these parameters should be computed.')
                            else:
                                break
                        else:
                            sg.PopupError('ERROR\nUser selected to <Compute reaction force and distance>\nHowever, neither option:\n\t<Compute Reaction Force> and/or <Compute Reaction Distance>\nwas selected under <Reaction Force and Distance Parameters>')

                    else:
                        # Make sure that a function is being run
                        if len(order) == 0 and ForceFunctionSelection:
                            sg.PopupError('ERROR\nUser has imported a file, but has not selected any processing functions to run on this file.')
                        else:
                            value.update({'order':order})
                            break
        # RUN0 corresponds to filtering parameters
        if event == '-RUN0-':
            isDisabled['FILT'] = not value['-RUN0-']
            window['MinDataRatio'].update(disabled=isDisabled['FILT'], text_color=tColors[int(value['-RUN0-'])])
            window['RTThreshold'].update(disabled=isDisabled['FILT'], text_color=tColors[int(value['-RUN0-'])])
            window['T_MinDataRatio'].update(text_color=bColors[int(value['-RUN0-'])])
            window['T_RTThreshold'].update(text_color=bColors[int(value['-RUN0-'])])
            window['U_MinDataRatio'].update(text_color=bColors[int(value['-RUN0-'])])
            window['U_RTThreshold'].update(text_color=bColors[int(value['-RUN0-'])])
            window['RemoveRT'].update(disabled=isDisabled['FILT'])
            window['RemoveAcc'].update(disabled=isDisabled['FILT'])
            window['CalibrateAcc'].update(disabled=isDisabled['FILT'])
            window['RemoveIncorrectResponses'].update(disabled=isDisabled['FILT'])
        # RUN1 corresponds to the reaction time parameters
        if event == '-RUN1-':
            isDisabled['RT'] = not value['-RUN1-']
            window['RTResLB'].update(disabled=isDisabled['RT'], text_color=tColors[int(value['-RUN1-'])])
            window['RTPeakDistance'].update(disabled=isDisabled['RT'], text_color=tColors[int(value['-RUN1-'])])
            window['T_RTResLB'].update(text_color=bColors[int(value['-RUN1-'])])
            window['T_RTPeakDistance'].update(text_color=bColors[int(value['-RUN1-'])])
            window['U_RTResLB'].update(text_color=bColors[int(value['-RUN1-'])])
            window['U_RTPeakDistance'].update(text_color=bColors[int(value['-RUN1-'])])
        # RUN2 corresponds to the rotation correction
        if event == '-RUN2-':
            isDisabled['ROTCORRECT'] = not value['-RUN2-']
            window['StoreTheta'].update(disabled=isDisabled['ROTCORRECT'])
        # RUN3 corresponds to the distance computations
        if event == '-RUN3-':
            isDisabled['DISTANCE'] = not value['-RUN3-']
            window['ComputeTotalDistance'].update(disabled=isDisabled['DISTANCE'])
        # RUN4 corresponds to the peak acceleration and distance computations
        if event == '-RUN4-':
            isDisabled['RFRD'] = not value['-RUN4-']
            window['-X-'].update(disabled=isDisabled['RFRD'])
            window['-Y-'].update(disabled=isDisabled['RFRD'])
            window['-Z-'].update(disabled=isDisabled['RFRD'])            
            window['absolute'].update(disabled=isDisabled['RFRD'])
            window['-RF-'].update(disabled=isDisabled['RFRD'])
            window['-RD-'].update(disabled=isDisabled['RFRD'])
        

    window.close()
    if value is not None:
        value.update({'axes':MapAxes([value['-X-'], value['-Y-'], value['-Z-']])})
        which = ['acceleration', 'distance']
        which = [which[i] for i, val in enumerate([value['-RF-'], value['-RD-']]) if val]
        value.update({'which':which})
        value.update({'OldTagList':oldOrder})

    if cancelled:
        value = None

    return value



def AnalysisParams(Title, Data, config = None, ImportFilePath = None):
    
    def GroupStimSetWindow(Sessions, stimuli, values, Mapping = None, MasterWindow = True, Location = None):
        MAP = {'+':True, '-':False}
        INVMAP = {not v:k for k, v in MAP.items()}
        StimSetGrouping = []
        N = len(Sessions) * len(stimuli) * 2
        if N > 16:
            sessionLen = Sessions[0:4]
            stimLen = stimuli[0:2]
        else:
            sessionLen = Sessions
            stimLen = stimuli
        for sessIdx, session in enumerate(sessionLen):
            for stimIdx, stim in enumerate(stimLen):
                row = [
                        [sg.Column([[sg.Button(button_text = INVMAP[not bool(stimIdx)], enable_events = True, key = '-ADD_{}_{}-'.format(sessIdx, stimIdx))]]),
                        sg.Column([[sg.Text('Combine data from: ')]]),
                        sg.Column([[sg.Combo(values=list(Sessions), default_value = session, readonly = True, key = '-SESSION_{}_{}-'.format(sessIdx, stimIdx), visible = not bool(stimIdx))]]),
                        sg.Column([[sg.Text('with: ', visible = not bool(stimIdx), key='-TEXT1_{}_{}-'.format(sessIdx, stimIdx))]]),
                        sg.Column([[sg.Combo(values=stimuli, default_value = stim, readonly = True, key = '-STIM_{}_{}-'.format(sessIdx, stimIdx), visible = not bool(stimIdx))]]),
                        sg.Column([[sg.Text('into set: ', visible = not bool(stimIdx), key='-TEXT2_{}_{}-'.format(sessIdx, stimIdx))]]),
                        sg.Column([[sg.Combo(values = [values['-CONTROL-'], values['-TARGET-']], readonly=True, key = '-GRP_{}_{}-'.format(sessIdx, stimIdx), visible = not bool(stimIdx))]])]
                    ]
                StimSetGrouping.append([sg.Column(row, key = '-ROW_{}_{}-'.format(sessIdx, stimIdx))])
        
        StimSetGrouping.append([sg.Column([[sg.Button(button_text = 'Add more entries', enable_events = True, key = '-NEW_WINDOW-')]])])

        layout = [[sg.Frame('Please indicate the group allocation', StimSetGrouping)],
                  [sg.Cancel(key='-CANCEL-'), sg.Button(button_text = 'Confirm', enable_events = True, key = '-CONFIRM-')]]

        if Location is None:
            win = sg.Window('Group allocation', layout)
        else:
            win = sg.Window('Group allocation', layout, location=Location)

        cancelled = False
        while True:
            e, v = win.read()
            if e in (sg.WIN_CLOSED, '-CANCEL-'):
                cancelled = True
                break
            if e.startswith('-ADD'):
                ID = e.split('-ADD')[-1]
                txt = win[e].GetText()
                win[e].update(text = INVMAP[MAP[txt]])
                win['-SESSION{}'.format(ID)].update(visible = MAP[txt])
                win['-TEXT1{}'.format(ID)].update(visible = MAP[txt])
                win['-STIM{}'.format(ID)].update(visible = MAP[txt])
                win['-TEXT2{}'.format(ID)].update(visible = MAP[txt])
                win['-GRP{}'.format(ID)].update(visible = MAP[txt])
            if e == '-NEW_WINDOW-':
                loc = win.CurrentLocation()
                Mapping = GroupStimSetWindow(Sessions, stimuli, values, Mapping = Mapping, MasterWindow = False, Location = (loc[0] + 10, loc[1] + 10) )
            if e == '-CONFIRM-':
                if Mapping is None:
                    Mapping = {}
                IDDict = {}
                IDs = []
                cond = {values['-CONTROL-']:False, values['-TARGET-']:False}
                for key in v.keys():
                    if key.startswith('-SESSION'):
                        ID = key.split('-SESSION')[-1]
                        IDs.append(ID)
                        # Store which ID the key belongs to 
                        IDDict.update({ID:v[key]})

                for ID in IDs:
                    key = '-ADD{}'.format(ID)
                    if not MAP[win[key].GetText()]:
                        if v['-SESSION{}'.format(ID)] in Mapping.keys():
                            sessDict = Mapping[v['-SESSION{}'.format(ID)]]
                        else:
                            sessDict = {}
                        sessDict.update({v['-STIM{}'.format(ID)]:v['-GRP{}'.format(ID)]})
                        Mapping.update({v['-SESSION{}'.format(ID)]:sessDict})

                # Probably a better way to do this
                for sess in Mapping.keys():
                    if values['-CONTROL-'] in Mapping[sess].values():
                        cond.update({values['-CONTROL-']:True})
                    if values['-TARGET-'] in Mapping[sess].values():
                        cond.update({values['-TARGET-']:True})

                if all(cond[key] for key in cond.keys()):
                    break
                else:
                    if MasterWindow:
                        sg.PopupError('ERROR \n Please make sure that at least one group is allocated to the control: <{}> \n and one group is allocated to the target: <{}>.'.format(values['-CONTROL-'], values['-TARGET-']))
                    else:
                        break

        win.close()
        if cancelled:
            Mapping = None

        return Mapping
    

    def CheckCompatibility(FilePath):
        MetaDataFilepath = FilePath[0:-4] + "_metadata.json"
        IsCompatible = False
        if os.path.isfile(MetaDataFilepath):
            with open(MetaDataFilepath, 'r', encoding = 'utf-8') as f:
                jsonfile = json.loads(f.read(), strict = False)
            f.close()
            try:
                if 'RESAMPLE' in jsonfile['TagList']:
                    IsCompatible = True
                else:
                    sg.PopupError('ERROR\n Cannot analyse the imported file: \n {} \n since filtering (and resampling) has not been performed.'.format(MetaDataFilepath))
            except KeyError:
                sg.PopupError('ERROR\n Invalid metadata file in \n {} \n Cannot determine what processing has been conducted'.format(MetaDataFilepath))
        else:
            sg.PopupError('ERROR\n Metadata file cannot be found at expected location: \n {}'.format(MetaDataFilepath))
        return IsCompatible
    

    if CheckCompatibility(ImportFilePath):
        # Default summary options
        PreProLabel = 'Preprocessed data'
        WithinLabel = 'Within participant averaged data'
        TarOptions = {True:[PreProLabel, WithinLabel], False:[PreProLabel]}
        DefaultSummaryOptions = {'-WITHIN_AVG-':False,
                                '-RESTRUCTURE-':False,
                                '-BETWEEN_AVG-':False,
                                '-GEN_STATS-':False,
                                '-LMM-':False,
                                '-GEN_STATS_TAR-':TarOptions[True],
                                '-LMM_TAR-':TarOptions[True]}

        # Infer the Control and Target stimuli
        DataImporter = AAT.DataImporter('Dummy', 'Dummy', printINFO=False)
        Sessions = np.unique(Data[DataImporter.constants['SESSION_COLUMN']].to_numpy())
        stimSets = np.unique(Data[DataImporter.constants['STIMULUS_SET_COLUMN']].to_numpy())
        stimuli = []
        # Remove practice stimuli names
        for stim in stimSets:
            if not 'practice' in stim.lower():
                stimuli.append(stim)

        DefaultSummaryOptions.update({'-CONTROL-':stimuli[0], '-TARGET-':stimuli[1]})

        if config:
            # Analysis options config
            configSummary = config['Analysis Parameters']
            DefaultSummaryOptions.update({'-WITHIN_AVG-':configSummary['Averaging Options']['Compute within participant averages'],
                                        '-RESTRUCTURE-':configSummary['Averaging Options']['Restructure averaged data for LMM'],
                                        '-BETWEEN_AVG-':configSummary['Averaging Options']['Compute between participant averages'],
                                        '-GEN_STATS-':configSummary['Statistics']['Compute means and standard deviations'],
                                        '-LMM-':configSummary['Statistics']['Compute Linear Mixed Models'],
                                        '-CONTROL-':configSummary['Control'],
                                        '-TARGET-':configSummary['Target'],
                                        '-GEN_STATS_TAR-':TarOptions[configSummary['Averaging Options']['Compute within participant averages']],
                                        '-LMM_TAR-':TarOptions[configSummary['Averaging Options']['Compute within participant averages']]})


        # ANALYSIS OPTIONS LAYOUT

        StimuliLayout = [   
                            [sg.Column([
                                [sg.Text('Control Stimulus')],
                                [sg.Text('Target Stimulus')]
                            ]),
                            sg.Column([
                                [sg.Combo(values=stimuli, default_value=DefaultSummaryOptions['-CONTROL-'], key='-CONTROL-', size = (20,1))],
                                [sg.Combo(values=stimuli, default_value=DefaultSummaryOptions['-TARGET-'], key='-TARGET-', size = (20,1))]
                            ])],
                            [sg.Button(button_text='Specify stimuli set allocation', enable_events = True, key='-SPECIFY_STIM_SET-')]
                        ]

        AveragingLayout = [
                            [sg.Checkbox('Compute averages within participant', key = '-WITHIN_AVG-', default = DefaultSummaryOptions['-WITHIN_AVG-'], enable_events = True, tooltip='Averages over trials, of the same condition, for a given participant')],
                            [sg.Checkbox('Restructure witin participant averages for LMM', key = '-RESTRUCTURE-', default = DefaultSummaryOptions['-RESTRUCTURE-'], disabled = not DefaultSummaryOptions['-WITHIN_AVG-'])],
                            [sg.Checkbox('Compute averages between participant', key = '-BETWEEN_AVG-', default = DefaultSummaryOptions['-BETWEEN_AVG-'], disabled = not DefaultSummaryOptions['-WITHIN_AVG-'], tooltip='Averages, over the same conditions, across participants')]
                        ]

        StatisticsLayout = [
                            [sg.Column([
                                [sg.Checkbox('Compute means and standard deviations', key = '-GEN_STATS-', default=DefaultSummaryOptions['-GEN_STATS-'])],
                                [sg.Checkbox('Compute Linear Mixed Model (LMM)', key = '-LMM-', default=DefaultSummaryOptions['-LMM-'])]
                            ]),
                            sg.Column([
                                [sg.Text('from')],
                                [sg.Text('from')]
                            ]),
                            sg.Column([
                                [sg.Combo(DefaultSummaryOptions['-GEN_STATS_TAR-'], default_value=PreProLabel, key = '-GEN_STATS_TAR-', readonly = True, size=(30, 1))],
                                [sg.Combo(DefaultSummaryOptions['-LMM_TAR-'], default_value=PreProLabel, key = '-LMM_TAR-', readonly = True, size=(30, 1))],
                            ])
                            ]
                        ]

        AnalysisParamsLayout = [
                                [sg.Text('')],
                                [sg.Frame('Stimuli', StimuliLayout, relief=sg.RELIEF_SUNKEN)],
                                [sg.Frame('Averaging Options', AveragingLayout, relief=sg.RELIEF_SUNKEN)],
                                [sg.Frame('Statistics', StatisticsLayout, relief=sg.RELIEF_SUNKEN)]
                            ]                   


        layout = [
                    [sg.TabGroup([[sg.Tab('AnalysisParameters', AnalysisParamsLayout, font = ('Helvetica', 12))]])],
                    [sg.Cancel(key = '-CANCEL-'), sg.Button('Next', key = '-NEXT-')]
                ]

        window = sg.Window(Title, layout)

        NecessaryParams = ['-WITHIN_AVG-', '-LMM-', '-GEN_STATS-']

        cancelled = False
        StimSetMapping = None
        while True:
            event, value = window.read()
            if event in (sg.WIN_CLOSED, '-CANCEL-'):
                cancelled = True
                break
            if event == '-NEXT-':
                if all(not value[i] for i in NecessaryParams):
                    sg.PopupError('ERROR\n User indicated to run analysis, but no analysis functions were selected!')
                    # Check if the chosen stimuli exist in the dataframe
                else:
                    if StimSetMapping is not None:
                        if value['-LMM-']:
                            if value['-LMM_TAR-'] == WithinLabel and not value['-RESTRUCTURE-']:
                                sg.PopupError('ERROR\n User opted to compute LMM on the within participant averaged data\n However, the option to \n\t <Restructure within participant averages for LMM> \n is unselected.')
                            else:
                                break
                        else:
                            break
                    else:
                        sg.PopupError('ERROR\n Mapping of stimulus sets to control: {} and to target: {} has not been specified.\n Please click <Specify stimuli set allocation>'.format(value['-CONTROL-'], value['-TARGET-']))
            if event == '-WITHIN_AVG-':
                window['-RESTRUCTURE-'].update(disabled = not value['-WITHIN_AVG-'])
                window['-BETWEEN_AVG-'].update(disabled = not value['-WITHIN_AVG-'])
                window['-GEN_STATS_TAR-'].update(value = PreProLabel, values=TarOptions[value['-WITHIN_AVG-']])
                window['-LMM_TAR-'].update(value = PreProLabel, values=TarOptions[value['-WITHIN_AVG-']])
                if not value['-WITHIN_AVG-']:
                    window['-RESTRUCTURE-'].update(value=False)
                    window['-BETWEEN_AVG-'].update(value=False)
            if event == '-SPECIFY_STIM_SET-':
                if any(not len(value[i]) > 0 for i in ['-CONTROL-', '-TARGET-']):
                    sg.PopupError('ERROR\n Please specify a Control and a Target stimuli.')
                elif value['-CONTROL-'] == value['-TARGET-']:
                    sg.PopupError('ERROR\n The Control cannot be the same as the Target')
                else:
                    window.hide()
                    StimSetMapping = GroupStimSetWindow(Sessions, stimuli, value, Mapping = StimSetMapping)
                    window.UnHide()

        window.close()
        
        # Store data targets
        # Note, -WITHIN_AVG- is not stored here since we
        # handle it differently in RunAnalysis, since other
        # functions may depend on it 
        DataTars = {PreProLabel:[], WithinLabel:[]}
        AvgFuncs = ['-RESTRUCTURE-', '-BETWEEN_AVG-']
        StatsFuncs = ['-GEN_STATS-', '-LMM-']
        
        for func in AvgFuncs:
            if value[func]:
                _TarList = DataTars[WithinLabel]
                _TarList.append(func)
                DataTars.update({WithinLabel:_TarList})
        
        for func in StatsFuncs:
            if value[func]:
                _TarList = DataTars[value[func[:-1]+'_TAR-']]
                _TarList.append(func)
                DataTars.update({value[func[:-1]+'_TAR-']:_TarList})

        value.update({'DataTargets':DataTars})
        value.update({'Mapping':StimSetMapping})
    
    else:
        cancelled = True

    if cancelled:
        value = None

    return value



def RunPreprocessing(Title, Params, Files, SavePath, SaveParams = None):

    def Proceed():
        output.print('[ DONE ]', background_color = 'green', text_color = 'white')
        window['-NEXT-'].update(disabled=False)
        return None

    def Process(TagID, InputData, Parameters, sgParameters):
        OutData = None
        if TagID == 'IMPORT':
            OutData = DataImporter.ImportData(sgParams=sgParameters)
            window.refresh()
        elif TagID == 'NANFILT':
            OutData = DataImporter.FilterNaNs(InputData, sgParams=sgParams)
            window.refresh()
        elif TagID == 'RT':
            OutData = DataImporter.ComputeRT(InputData, RTResLB = float(Parameters['RTResLB']), 
                                            RTPeakDistance = float(Parameters['RTPeakDistance']), 
                                            sgParams=sgParameters)
            window.refresh()
        elif TagID == 'FILT':
            OutData, _ = DataImporter.FilterData(InputData, RemoveIncorrectResponses = Parameters['RemoveIncorrectResponses'], MinDataRatio = float(Parameters['MinDataRatio']),
                                        RemoveRT = Parameters['RemoveRT'], RTThreshold = float(Parameters['RTThreshold']),
                                        CalibrateAcc = Parameters['CalibrateAcc'], RemoveAcc = Parameters['RemoveAcc'],
                                        sgParams = sgParameters)
            window.refresh()
        elif TagID == 'RESAMPLE':
            OutData = DataImporter.ResampleData(InputData, sgParams = sgParameters)
            window.refresh()
        elif TagID == 'ROTCORRECT':
            OutData = DataImporter.Correct4Rotations(InputData, StoreTheta = Parameters['StoreTheta'], sgParams=sgParameters)
            window.refresh()
        elif TagID == 'DISTANCE':
            OutData = DataImporter.ComputeDistance(InputData, ComputeTotalDistance = Params['ComputeTotalDistance'], sgParams=sgParameters)
            window.refresh()
        elif TagID == 'RFRD':
            OutData = DataImporter.ComputeDeltaAandD(InputData, axes = Parameters['axes'], absolute = Parameters['absolute'], which = Parameters['which'], sgParams = sgParameters)
            window.refresh()

        return OutData, sgParameters

    def CheckFile(Filename, SavePath, counter = 0):
        findingName = True
        fname = Filename
        while findingName:
            if not os.path.isfile(os.path.join(SavePath, '{}.pkl'.format(fname))) and not os.path.isfile(os.path.join(SavePath, '{}.csv'.format(fname))):
                findingName = False
            else:
                fname = Filename + '_{}'.format(counter)
            counter += 1

        return fname

    def SaveMetaData(filename, filetype, savepath, order, params):
        if 'OldTagList' in params.keys():
            order = params['OldTagList'] + order
        MetaData = {'TagList':order, 'Tag':order[-1], 'FileType':filetype, 'Params':params}

        with open(os.path.join(savepath, filename), 'w') as f:
            json.dump(MetaData, f)

        return None

    Data = None

    if len(Files) == 2:
        DataPath = Files[0]
        CondPath = Files[1]
        RunFromFile = False
        DataImporter = AAT.DataImporter(CondPath, DataPath, printINFO=False)

    elif len(Files) == 1:
        FilePath = Files[0]
        RunFromFile = True
        DataImporter = AAT.DataImporter('_', '_', printINFO=False)
    
    if RunFromFile:
        RunList = Params['order']
    else:
        RunList = ['IMPORT', 'NANFILT']
        for func in Params['order']:
            RunList.append(func)

    layout = [
        [sg.Text('Running AAT Processing')],
        [sg.T('0 %', key = '-PERCENTAGE-', size = (5,1)), sg.ProgressBar(100, orientation='h', size = (50, 20), key='-PROGRESS_BAR-'), 
              sg.T('Time remaining = 0 [s]', key = '-TIME-', size = (20, 1))],
        [sg.Multiline(size = (106, 15), key = '-OUTPUT-', autoscroll = True, disabled=True)],
        [sg.Cancel(key='-CANCEL-'), sg.Button('Run', key = '-RUN-'), sg.Button('Next', key = '-NEXT-', disabled=True)]
    ]
    
    window = sg.Window(Title, layout)

    percentage = window.FindElement('-PERCENTAGE-')
    prog_bar = window.FindElement('-PROGRESS_BAR-')
    time = window.FindElement('-TIME-')
    output = window.FindElement('-OUTPUT-')

    sgParams = {'WINDOW':window, 'OUTPUT':output, 'PERCENTAGE':percentage, 
                'PROG_BAR':prog_bar, 'TIME':time, 'IS_CLOSED':False}

    cancelled = False
    while not cancelled:
        event, value = window.read()
        if event in (sg.WIN_CLOSED, '-CANCEL-'):
            cancelled = True
            break
        if event == '-NEXT-':
            break
        if event == '-RUN-':
            window['-RUN-'].update(disabled=True)

            if RunFromFile:
                window['-OUTPUT-'].print('[ INFO ] Loading data from file, this may take a while and the program may (briefly) appear to be not responding. This is normal, please wait...')
                window.refresh()
                Data = DataImporter.LoadDF(FilePath, None)    
            
            for runFunc in RunList:
                Data, sgParams = Process(runFunc, Data, Params, sgParams)
                if sgParams['IS_CLOSED']:
                    output.print('[ ERROR ] User cancelled process.', background_color='red', text_color = 'white')
                    cancelled = True
                    break

            if SavePath is not None:
                Filetype = "pkl"
                if SaveParams is not None:
                    if SaveParams['-SMALLSAVE-']:
                        if SaveParams['-SAVECSV-']:
                            Filetype = 'csv'
                        CondensedData = DataImporter.CondenseDF(Data, sgParams = sgParams)
        
                Filename = CheckFile(SaveParams['-SAVE_FILENAME-'], SavePath)
                # window['-OUTPUT-'].print('[ INFO ] Filename: {}'.format(Filename))
                window['-OUTPUT-'].print('[ INFO ] Saving processed file to {}, this may take a while and the program may (briefly) appear to be not responding. This is normal, please wait...'.format(SavePath))
                window.refresh()
                MetaDataFile = '{}_metadata.json'.format(Filename)
                if SaveParams['-SAVECSV-']:
                    DataImporter.SaveDF('{}.{}'.format(Filename, Filetype), CondensedData, SavePath)
                else:
                    DataImporter.SaveDF('{}.{}'.format(Filename, Filetype), Data, SavePath)
                SaveMetaData(MetaDataFile, Filetype, SavePath, RunList, Params)
                SaveParams.update({'Processed_FilePath':os.path.join(SavePath, '{}.{}'.format(Filename, Filetype))})
            if not cancelled:
                Proceed()
            
    window.close()

    return Data



def RunAnalysis(Title, Data, Params, DataFilePath = None, SaveParams = None):
        
    def Proceed():
        output.print('[ DONE ]', background_color = 'green', text_color = 'white')
        window['-NEXT-'].update(disabled=False)
        return None


    def CheckFile(Filename, SavePath, counter = 0):
        findingName = True
        fname = Filename
        while findingName:
            if not os.path.isfile(os.path.join(SavePath, '{}.pkl'.format(fname))) and not os.path.isfile(os.path.join(SavePath, '{}.csv'.format(fname))) and not os.path.isfile(os.path.join(SavePath, '{}.xlsx'.format(fname))):
                findingName = False
            else:
                fname = Filename + '_{}'.format(counter)
            counter += 1

        return fname


    def Process(funcLabel, data, SaveParams, sgParams, returnData = False):
        if funcLabel == '-WITHIN_AVG-':
            D = Analysis.AverageWithinParticipant(data, sgParams = sgParams)
            if SaveParams['-SAVE-']:
                Filename = SaveParams['Analysis_Filename'] + '_Averaged_Within'
                Filename = CheckFile(Filename, SaveParams['Analysis_Filepath'])
                Filename = Filename + '.pkl'
                sgParams['OUTPUT'].print('[ INFO ] Saving averaged within participants data to {}'.format(SaveParams['Analysis_Filepath']))
                D.to_pickle(os.path.join(SaveParams['Analysis_Filepath'], Filename))
                sgParams['OUTPUT'].print('[ INFO ] \tSuccessfully saved as {}!'.format(Filename))
        elif funcLabel == '-RESTRUCTURE-':
            D = Analysis.RestructureAvgDF(data, sgParams = sgParams)
        elif funcLabel == '-BETWEEN_AVG-':
            D = Analysis.AverageBetweenParticipant(data, sgParams = sgParams)
            if SaveParams['-SAVE-']:
                Filename = SaveParams['Analysis_Filename'] + '_Averaged_Between'
                Filename = CheckFile(Filename, SaveParams['Analysis_Filepath'])
                Filename = Filename + '.pkl'
                sgParams['OUTPUT'].print('[ INFO ] Saving averaged between participants data to {}'.format(SaveParams['Analysis_Filepath']))                    
                D.to_pickle(os.path.join(SaveParams['Analysis_Filepath'], Filename))
                sgParams['OUTPUT'].print('[ INFO ] \tSuccessfully saved as {}!'.format(Filename))    
        elif funcLabel == '-GEN_STATS-':
            D = Analysis.GeneralStats(data, sgParams = sgParams)
            for metric in D.keys():
                Filename = SaveParams['Analysis_Filename'] + '_General_Statistics_{}'.format(metric)
                Filename = CheckFile(Filename, SaveParams['Analysis_Filepath'])
                Filename = Filename + '.csv'
                sgParams['OUTPUT'].print('[ INFO ] Saving general statistics {} data to {}'.format(metric, SaveParams['Analysis_Filepath']))
                D[metric].to_csv(os.path.join(SaveParams['Analysis_Filepath'], Filename))
                sgParams['OUTPUT'].print('[ INFO ] \tSuccessfully saved as {}!'.format(Filename))                
        elif funcLabel == '-LMM-':
            Map = {'rt':'RT', 'mda':'Acceleration', 'mdd':'Distance'}
            D, mdl = Analysis.LinearMixedModels(data, ReturnModel = True, sgParams = sgParams)

            for metric in mdl.keys():
                mdf = mdl[metric].fit()
                Filename = SaveParams['Analysis_Filename'] + '_LMM_Results_{}'.format(Map[metric])
                Filename = CheckFile(Filename, SaveParams['Analysis_Filepath'])
                Filename = Filename + '.csv'
                sgParams['OUTPUT'].print('[ INFO ] Saving LMM {} data to {}'.format(metric, SaveParams['Analysis_Filepath']))
                mdf.summary().tables[1].to_csv(os.path.join(SaveParams['Analysis_Filepath'], Filename))
                sgParams['OUTPUT'].print('[ INFO ] \tSuccessfully saved as {}!'.format(Filename))                
        
        if returnData:
            outData = D
        else:
            outData = None
        return outData, sgParams


    def GetFileName(saveparams):
        if saveparams['-SAVE-']:
            Filename = saveparams['-SAVE_FILENAME-']
            Filepath = saveparams['-SAVE_DIR-']
        else:
            try:
                FilePathList = saveparams['-FILE_PATH-'].split('/')
                if len(FilePathList) <= 1:
                    FilePathList = saveparams['-FILE_PATH-'].split('\\')
                    if len(FilePathList) <= 1:
                        FilePathList = os.getcwd().split('\\')
                        FilePathList.append('AAT_Analysis_Data.pkl')
                Filename = FilePathList[-1][0:-4]
                Filepath = '\\'.join(FilePathList[0:-1])
            except KeyError:
                Filename = 'AAT_Analysis_Data'
                Filepath = os.getcwd()
            
        return Filename, Filepath


    def SaveMapping(Mapping, saveparams):
        ColorMap = {}
        Colors = ['#FF6666', '#FFB266', '#FFFF66', '#B2FF66', '#66FFFF', '#66B2FF', '#B266FF', '#FF66B2', '#C0C0C0', '#FFFFFF',
                  '#FFCCCC', '#FFE5CC', '#FFFFCC', '#E5FFCC', '#CCFFFF', '#CCE5FF', '#E5CCFF', '#FF66B2', '#606060', '#A0A0A0']
        N = len(Colors)
        i = 0
        for session in Mapping.keys():
            for stimset in Mapping[session].keys():
                if session not in ColorMap.keys():
                    ColorMap.update({session:Colors[int(i % N)]})
                    i += 1
                if stimset not in ColorMap.keys():
                    ColorMap.update({stimset:Colors[int(i % N)]})
                    i += 1
                if Mapping[session][stimset] not in ColorMap.keys():
                    ColorMap.update({Mapping[session][stimset]:Colors[int(i % N)]})
                    i += 1
        
        Filename = saveparams['Analysis_Filename'] + '_GroupMapping'
        Filename = CheckFile(Filename, saveparams['Analysis_Filepath'])
        Filename = Filename + ".xlsx"
        Workbook = xls.Workbook(os.path.join(saveparams['Analysis_Filepath'], Filename))
        worksheet = Workbook.add_worksheet('Stimulus set mapping')
        bold = Workbook.add_format({'bold':True})
        worksheet.write(0, 0, 'Session', bold)
        worksheet.write(0, 1, 'Stimulus Set', bold)
        worksheet.write(0, 2, 'Control/Target Group', bold)
        row = 1
        for session in Mapping.keys():
            for stimset in Mapping[session].keys():
                colors = Workbook.add_format()
                colors.set_bg_color(ColorMap[session])
                worksheet.write(row, 0, session, colors)
                colors = Workbook.add_format()
                colors.set_bg_color(ColorMap[stimset])
                worksheet.write(row, 1, stimset, colors)
                colors = Workbook.add_format()
                colors.set_bg_color(ColorMap[Mapping[session][stimset]])
                worksheet.write(row, 2, Mapping[session][stimset], colors)
                row+=1

        Workbook.close()

        return None


    DataImporter = AAT.DataImporter('Dummy', 'Dummy', printINFO=False)
    Analysis = AAT.Analysis(Params['-CONTROL-'], Params['-TARGET-'], constants = DataImporter.constants, printINFO=False)

    layout = [
        [sg.Text('Running AAT Analysis')],
        [sg.T('0 %', key = '-PERCENTAGE-', size = (5,1)), sg.ProgressBar(100, orientation='h', size = (50, 20), key='-PROGRESS_BAR-'), 
              sg.T('Time remaining = 0 [s]', key = '-TIME-', size = (20, 1))],
        [sg.Multiline(size = (106, 15), key = '-OUTPUT-', autoscroll = True, disabled=True)],
        [sg.Cancel(key='-CANCEL-'), sg.Button('Run', key = '-RUN-'), sg.Button('Next', key = '-NEXT-', disabled=True)]
    ]
    
    window = sg.Window(Title, layout)

    percentage = window.FindElement('-PERCENTAGE-')
    prog_bar = window.FindElement('-PROGRESS_BAR-')
    time = window.FindElement('-TIME-')
    output = window.FindElement('-OUTPUT-')

    sgParams = {'WINDOW':window, 'OUTPUT':output, 'PERCENTAGE':percentage, 
                'PROG_BAR':prog_bar, 'TIME':time, 'IS_CLOSED':False}

    cancelled = False
    while not cancelled:
        event, value = window.read()
        if event in (sg.WIN_CLOSED, '-CANCEL-'):
            cancelled = True
            break
        if event == '-NEXT-':
            break
        if event == '-RUN-':
            window['-RUN-'].update(disabled=True)

            Filename, Filepath = GetFileName(SaveParams)
            SaveParams.update({'Analysis_Filename':Filename, 'Analysis_Filepath':Filepath})

            if SaveParams['-SAVE-']:
                SaveMapping(Params['Mapping'], SaveParams)

            window['-OUTPUT-'].print('[ INFO ] Applying stimulus set mapping...')
            Data = Analysis.FuseStimulusSets(Data, Params['Mapping'], sgParams = sgParams)
            window.refresh()
            
            if len(Params['DataTargets']['Preprocessed data']) > 0:
                for func in Params['DataTargets']['Preprocessed data']:
                    _, sgParams = Process(func, Data, SaveParams, sgParams)
                    window.refresh()
                    if sgParams['IS_CLOSED']:
                        output.print('[ ERROR ] User cancelled process.', background_color='red', text_color = 'white')
                        cancelled = True
                        break

            if Params['-WITHIN_AVG-']:
                Data, sgParams = Process('-WITHIN_AVG-', Data, SaveParams, sgParams, returnData = True)
                if sgParams['IS_CLOSED']:
                    output.print('[ ERROR ] User cancelled process.', background_color='red', text_color = 'white')
                    cancelled = True
                    break

                if len(Params['DataTargets']['Within participant averaged data']) > 0:
                    for func in Params['DataTargets']['Within participant averaged data']:
                        if func == '-RESTRUCTURE-':
                            ResData, sgParams = Process(func, Data, SaveParams, sgParams, returnData = True)
                        elif func == '-LMM-':
                            _, sgParams = Process(func, ResData, SaveParams, sgParams)
                        else:
                            _, sgParams = Process(func, Data, SaveParams, sgParams)
                        window.refresh()
                        if sgParams['IS_CLOSED']:
                            output.print('[ ERROR ] User cancelled process.', background_color='red', text_color = 'white')
                            cancelled = True
                            break

            if not cancelled:
                Proceed()
            
    window.close()

    return None


#=======================================================================================================================#

Title = 'AAT Analysis'

configPath = os.path.join(os.getcwd(), 'GUIConfig.json')
if os.path.isfile(configPath):
    with open(configPath, 'r', encoding = 'utf-8') as f:
        configFile = json.loads(f.read(), strict = False)
    f.close()
    if not configFile['UseConfig']:
        configFile = None
else:
    configFile = None

RunAll = StartWindow(Title, config=configFile)

if RunAll is not None:
    # If user selected to import from raw data (for both processing only or processing + analysis)
    if RunAll in (0, 1):
        PreFolderPaths = PreProcessFolder(Title, config=configFile)
        if PreFolderPaths:
            RawDataPath = PreFolderPaths['-RAW_DATA_DIR-']
            CondDataPath = PreFolderPaths['-COND_DATA_DIR-']
            PreSavePath = PreFolderPaths['-SAVE_DIR-']

            # Show screen for processing params
            UserInputs = Pre_FunctionParams(Title, config=configFile, ForAnalysis=bool(abs(RunAll - 1)))
            if UserInputs:
                Data = RunPreprocessing(Title, UserInputs, [RawDataPath, CondDataPath], PreSavePath, SaveParams=PreFolderPaths)
                if RunAll == 0:
                    AnalysisInputs = AnalysisParams(Title, Data, config=configFile, ImportFilePath=PreFolderPaths['Processed_FilePath'])
                    if AnalysisInputs:
                        RunAnalysis(Title, Data, AnalysisInputs, SaveParams=PreFolderPaths)
            else: 
                pass
    # If user selected to import from file
    elif RunAll in (2, 3):
        ImportFilePaths = LoadFromFile(Title, config = configFile)
        if ImportFilePaths:
            ImportedFile = ImportFilePaths['-FILE_PATH-']
            ImpSavePath = ImportFilePaths['-SAVE_DIR-']
            if RunAll == 2:
                UserInputs = Pre_FunctionParams(Title, config=configFile, ImportFilePath=ImportedFile)
                if UserInputs:
                    Data = RunPreprocessing(Title, UserInputs, [ImportedFile], ImpSavePath, SaveParams=ImportFilePaths)
            elif RunAll == 3:
                Data = LoadData(Title, ImportedFile)
                if Data is not None:
                    AnalysisInputs = AnalysisParams(Title, Data, config = configFile, ImportFilePath = ImportedFile)
                    if AnalysisInputs:
                        RunAnalysis(Title, Data, AnalysisInputs, SaveParams = ImportFilePaths)
                else: 
                    sg.PopupError('ERROR\n Failed to load data')
            

#=======================================================================================================================#