import PySimpleGUI as sg
import os
import numpy as np

import AAT as AAT

def LoadFromFile(Title, config = None):
    fieldEntrySize = (100, 1)

    defaults = {'-FILE_PATH-':''}
    isDisabled = True
    if config is not None:
        Dirs = config['Import File Information']
        defaults.update({'-FILE_PATH-':Dirs['File Path']})

    bColors = ['grey42', 'white']
    tColors = ['grey42', 'black']

    # Build window layout
    layout = [[sg.Text('Please locate the File Path: ')],
              [sg.In(default_text = defaults['-FILE_PATH-'], size=fieldEntrySize, key = '-FILE_PATH-', enable_events=True), sg.FileBrowse(file_types=(('Pickle Files', '*.pkl'),))],   
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
        else:
            if os.path.isfile(value['-FILE_PATH-']):
                window['-NEXT-'].update(disabled = False)
            else:
                window['-NEXT-'].update(disabled = True)            

    window.close()

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
        DataImporter = AAT.DataImporter('_', '_', printINFO=False)
        Data = DataImporter.LoadDF(FilePath, None)
        break

    window.close()

    if cancelled:
        Data = None    

    return Data


def PlotDefinition(Title, Data, SubplotLayout):

    def getFunctionOptionalParams(f):
        numInputArgs = f.__code__.co_argcount
        AllInputs = f.__code__.co_varnames[1:numInputArgs]
        fDefaults = f.__defaults__
        fArgs = AllInputs[-len(fDefaults):]
        OptionalParams = {k:fDefaults[v] for v,k in enumerate(fArgs)}
        return OptionalParams


    def SettingWindow():

        settings = {}

        rows = [
            [sg.Column([
                   [sg.Text('Number of rows: ')] 
            ]), 
            sg.Column([
                [sg.InputText(size = (3, 1), key = '-NUM_ROWS-', enable_events=True)]
            ])]
        ]

        cols = [
            [sg.Column([
                   [sg.Text('Number of columns: ')] 
            ]), 
            sg.Column([
                [sg.InputText(size = (3, 1), key = '-NUM_COLUMNS-', enable_events=True)]
            ])]
        ]

        layout = [
            [sg.Column(rows), sg.Column(cols)],
            [sg.Cancel(key='-SETTING_CANCEL-'), sg.Button('Apply', key = '-APPLY-', enable_events=True)]
        ]

        settingWindow = sg.Window('Plot Settings', layout)

        Scancelled = False
        while True:
            Sevent, Svalue = settingWindow.read()
            if Sevent in (sg.WIN_CLOSED, '-SETTING_CANCEL-'):
                Scancelled = True
                break
            if Sevent == '-APPLY-':
                try:
                    num_rows = int(Svalue['-NUM_ROWS-'])
                    num_columns = int(Svalue['-NUM_COLUMNS-'])
                    settings.update({'PlotLayout':(num_rows, num_columns)})
                    break
                except ValueError:
                    sg.PopupError('ERROR\n Invalid number of rows/columns, please input an integer.')

        settingWindow.close()
        if Scancelled:
            settings = None

        return settings


    def OptionWindow(Plot, PlotitngOptions, Stimuli, DF, oldParams = None):

        def isFloat(x, useInt = False):
            try:
                if not useInt:
                    out = isinstance(float(x), float)
                else:
                    out = isinstance(int(x), int)
            except ValueError:
                out = False
            return out

        
        defaults = {'axis':['Z'], 'movement':['Push', 'Pull'], 'Gradient':False, 
                    'ColorMap':'RdYlGn_r', 'Threshold':60, 'YLims':None, 
                    'ShowAxis':[True, True],'HideLegend':False}

        FuncDefaults = getFunctionOptionalParams(PlottingOptions[Plot])

        for key in FuncDefaults.keys():
            if key in defaults.keys():
                defaults.update({key:FuncDefaults[key]})

        SensorAxisLayout = [
            [sg.Frame('Sensor Components', [[sg.Column([[sg.Checkbox('X (Vertical Axis)', key = '-X-', enable_events = True, default='X' in defaults['axis'])]]),
                                sg.Column([[sg.Checkbox('Y (Lateral Axis)', key = '-Y-', enable_events = True, default='Y' in defaults['axis'])]]),
                                sg.Column([[sg.Checkbox('Z (Ventral Axis)', key = '-Z-', enable_events = True, default='Z' in defaults['axis'])]])]],
                                key = '-SENSOR_COMPONENTS-')]
        ]

        movementLayout = [
            [sg.Frame('Movement', [[sg.Column([[sg.Checkbox('Pull', key = '-PULL-', enable_events = True, default='Pull' in defaults['movement'])]]),
                                    sg.Column([[sg.Checkbox('Push', key = '-PUSH-', enable_events = True, default='Push' in defaults['movement'])]])]])]
        ]

        stimRow = []
        stimLim = 8
        for num, stim in enumerate(Stimuli):
            if num == stimLim:
                break
            vals = ['-'] + Stimuli
            row = sg.Column([[sg.Combo(vals, default_value = '-', enable_events=True, readonly=True, key = '-STIM_{}-'.format(stim))]])
            stimRow.append(row)

        stimulusLayout = [[sg.Frame('Stimulus', [stimRow])]]

        axisProperties = [
            [sg.Frame('Plot axis properties', [[sg.Column([[sg.Checkbox('Hide x-axis', enable_events = True, key = '-HIDE_X-', default=not defaults['ShowAxis'][0])]])],
                                               [sg.Column([[sg.Checkbox('Hide y-axis', enable_events = True, key = '-HIDE_Y-', default=not defaults['ShowAxis'][1])]])],
                                               [sg.Column([[sg.Checkbox('Specify x-limits', enable_events = True, key = '-XLIM-', default = False)]]),
                                                sg.Column([[sg.Text('min')]]),
                                                sg.Column([[sg.InputText(enable_events = True, key = '-XMIN-', size = (10, 1), disabled=True, use_readonly_for_disable=True)]]),
                                                sg.Column([[sg.Text('max')]]),
                                                sg.Column([[sg.InputText(enable_events = True, key = '-XMAX-', size = (10, 1), disabled=True, use_readonly_for_disable=True)]])],
                                               [sg.Column([[sg.Checkbox('Specify y-limits', enable_events = True, key = '-YLIM-', default = False)]]),
                                                sg.Column([[sg.Text('min')]]),
                                                sg.Column([[sg.InputText(enable_events = True, key = '-YMIN-', size = (10, 1), disabled=True, use_readonly_for_disable=True)]]),
                                                sg.Column([[sg.Text('max')]]),
                                                sg.Column([[sg.InputText(enable_events = True, key = '-YMAX-', size = (10, 1), disabled=True, use_readonly_for_disable=True)]])]])]
        ]

        plotProperties = [
            [sg.Frame('Plot Properties', [[sg.Column([[sg.Checkbox('Hide legend', enable_events = True, key = '-HIDE_LEGEND-', default = defaults['HideLegend'])]])],
                                          [sg.Column([[sg.Text('Font size')]]), sg.Column([[sg.InputText(default_text = '18', enable_events = True, key = '-FONTSIZE-', size=(10, 1))]])]])]
        ]

        metricLayout = [
            [sg.Frame('Metric', [[sg.Column([[sg.Combo(['acceleration', 'distance'], default_value = 'acceleration', key = '-METRIC-', enable_events = True, readonly = True)]])]])]
        ]

        colorMaps = ['RdYlGn_r', 'RdYlGn']
        gradientLayout = [
            [sg.Frame('Time encoding', [[sg.Column([[sg.Checkbox('Use gradient to indicate time', default = defaults['Gradient'], enable_events = True, key = '-GRADIENT-')]]), 
                                         sg.Column([[sg.Combo(colorMaps, default_value = defaults['ColorMap'], key = '-COLOR_MAP-', enable_events = True, disabled = not defaults['Gradient'], readonly=True)]])]])]
        ]
        
        # FunctionMap = {'Acceleration-Time':['SensorAxisLayout', 'MovementLayout', 'StimulusLayout', 'AxisProperties', 'PlotProperties'],
        #                'Distance-Time':['SensorAxisLayout', 'MovementLayout', 'StimulusLayout', 'AxisProperties', 'PlotProperties'],
        #                'Approach-Avoidance XZ':['MetricLayout', 'MovementLayout', 'StimulusLayout'],
        #                'Acceleration 3D':['GradientLayout', 'MovementLayout', 'StimulusLayout'],
        #                'Distance 3D':['GradientLayout', 'MovementLayout', 'StimulusLayout']}
        FunctionMap = {'Acceleration-Time':['SensorAxisLayout', 'MovementLayout', 'StimulusLayout', 'AxisProperties', 'PlotProperties'],
                       'Distance-Time':['SensorAxisLayout', 'MovementLayout', 'StimulusLayout', 'AxisProperties', 'PlotProperties'],
                       'Approach-Avoidance XZ':['MetricLayout', 'MovementLayout', 'StimulusLayout'],
                       'Acceleration 3D':['MovementLayout', 'StimulusLayout'],
                       'Distance 3D':['MovementLayout', 'StimulusLayout']}        
        LayoutMap = {'SensorAxisLayout':[sg.Column(SensorAxisLayout)],
                     'MovementLayout':[sg.Column(movementLayout)], 
                     'StimulusLayout':[sg.Column(stimulusLayout)], 
                     'AxisProperties':[sg.Column(axisProperties)], 
                     'PlotProperties':[sg.Column(plotProperties)],
                     'MetricLayout':[sg.Column(metricLayout)],
                     'GradientLayout':[sg.Column(gradientLayout)]}

        opt_layout = []
        for element in FunctionMap[Plot]:
            opt_layout.append(LayoutMap[element])

        Layout = [[sg.Text('Row '), sg.InputText(enable_events = True, key = '-P_NUM-', size = (10, 1))]] + opt_layout + [[sg.Cancel(key = '-CANCEL-'), sg.Button('Apply', key = '-OPT_APPLY-')]]

        win = sg.Window('Plot Options', Layout)

        PlotParams = {}
        
        cancelled = False
        while True:
            e, v = win.read()
            if e in (sg.WIN_CLOSED, '-CANCEL-'):
                cancelled = True
                break
            if e == '-OPT_APPLY-':
                isOkay = True
                if 'SensorAxisLayout' in FunctionMap[Plot]:
                    if all(not v[ax] for ax in ['-X-', '-Y-', '-Z-']):
                        isOkay = False
                        sg.PopupError('ERROR\n Please select at least one sensor axis to plot')
                    axes = ['X', 'Y', 'Z']
                    axes = [ax for ax in axes if v['-{}-'.format(ax)]]
                    PlotParams.update({'axis':axes})
                if 'MovementLayout' in FunctionMap[Plot]:
                    if all(not v[mov] for mov in ['-PULL-', '-PUSH-']):
                        isOkay = False
                        sg.PopupError('ERROR\n Please select at least one movement (Push and/or Pull) to plot')
                    movement = [mov for mov in ['Pull', 'Push'] if v['-{}-'.format(mov.upper())]]
                    PlotParams.update({'movement':movement})
                if 'StimulusLayout' in FunctionMap[Plot]:
                    if all(v[stim] == '-' for stim in ['-STIM_{}-'.format(s) for s in Stimuli]):
                        isOkay = False
                        sg.PopupError('ERROR\n Please select at least one stimulus to plot')
                    stimVals = []
                    for key in v.keys():
                        if key.startswith('-STIM_') and not v[key] == '-':
                            stimVals.append(v[key])
                    stims = list(np.unique(stimVals))
                    PlotParams.update({'stimulus':stims})
                if 'AxisProperties' in FunctionMap[Plot]:
                    if v['-XLIM-']:
                        if not isFloat(v['-XMIN-']) or not isFloat(v['-XMAX-']):
                            isOkay = False
                            sg.PopupError('ERROR\n Invalid x-axis limits. Please input a number.')
                        else:
                            PlotParams.update({'XLims':[float(v['-XMIN-']), float(v['-XMAX-'])]})
                    if v['-YLIM-']:
                        if not isFloat(v['-YMIN-']) or not isFloat(v['-YMAX-']):
                            isOkay = False
                            sg.PopupError('ERROR\n Invalid y-axis limits. Please input a number.')
                        else:
                            PlotParams.update({'YLims':[float(v['-YMIN-']), float(v['-YMAX-'])]})       
                    PlotParams.update({'ShowAxis':[not v['-HIDE_X-'], not v['-HIDE_Y-']]})
                if 'PlotProperties' in FunctionMap[Plot]:
                    PlotParams.update({'HideLegend':v['-HIDE_LEGEND-']})
                    if not isFloat(v['-FONTSIZE-'], useInt = True):
                        isOkay = False
                        sg.PopupError('ERROR\n Invalid Font size. Please input an integer.')
                    else:
                        Plotter.setFontSize(int(v['-FONTSIZE-']))
                if 'MetricLayout' in FunctionMap[Plot]:
                    PlotParams.update({'metric':v['-METRIC-']})
                if 'GradientLayout' in FunctionMap[Plot]:
                    PlotParams.update({'Gradient':v['-GRADIENT-']})
                    PlotParams.update({'ColorMap':v['-COLOR_MAP-']})

                if not isFloat(v['-P_NUM-'], useInt = True):
                    isOkay = False
                    sg.PopupError('ERROR\n Invalid row. Please specify a positive integer <= {}'.format(len(DF)))
                elif isFloat(v['-P_NUM-'], useInt = True):
                    if int(v['-P_NUM-']) > len(DF) or int(v['-P_NUM-']) < 0:
                        isOkay = False
                        sg.PopupError('ERROR\n Invalid row. Please specify a positive integer <= {}'.format(len(DF)))

                if isOkay:
                    PlotParams.update({'participant':int(v['-P_NUM-'])})
                    PlotParams.update({'DF':DF})
                    break
                
                pass
            if e == '-XLIM-':
                win['-XMIN-'].update(value = '', disabled=not v[e])
                win['-XMAX-'].update(value = '', disabled=not v[e])
            if e == '-YLIM-':
                win['-YMIN-'].update(value = '', disabled=not v[e])
                win['-YMAX-'].update(value = '', disabled=not v[e])
            if e == '-GRADIENT-':
                win['-COLOR_MAP-'].update(disabled = not v[e])

        win.close()

        if cancelled:
            PlotParams = None

        return PlotParams


    def GenLayout(Plots, subplotlayout):
        PlotNum = 1
        PlotLayout = []
        for row in range(subplotlayout[0]):
            rowLayout = []
            for col in range(subplotlayout[1]):
                InternalLayout = [[sg.Combo(Plots, default_value=Plots[0], enable_events=True, key='-PLOT_{}_{}-'.format(row, col), readonly=True), 
                                   sg.Button(button_text='Options', enable_events = True, key = '-OPTIONS_{}_{}-'.format(row, col), disabled=True)]]
                PlotFrame = sg.Frame('Plot {}'.format(PlotNum), InternalLayout, key='-FRAME_{}_{}-'.format(row, col), tooltip='No Data')
                element = sg.Column([[PlotFrame]])
                rowLayout.append(element)
                PlotNum += 1
            PlotLayout.append(rowLayout)
        
        BottomRow = [sg.Cancel(key='-CANCEL-'), sg.Button('Settings', key='-SETTINGS-', enable_events=True), 
                     sg.Button('Plot', key='-SHOW_PLOT-', enable_events = True)]
        
        PlotLayout.append(BottomRow)

        return PlotLayout


    Plotter = AAT.Plotter()
    # Derive the available stimuli for plotting
    Stimuli = []
    if Plotter.constants['STIMULUS_SET_COLUMN'] in Data.columns:
        Stimuli = list(np.unique(Data[Plotter.constants['STIMULUS_SET_COLUMN']]))
    else:
        for col in Data.columns:
            if col.startswith('RT '):
                Stimuli.append(col[8:])
        Stimuli = list(np.unique(Stimuli))

    
    PlottingOptions = {'None':None,
                       'Acceleration-Time':Plotter.AccelerationTime, 
                       'Distance-Time':Plotter.DistanceTime,
                       'Approach-Avoidance XZ':Plotter.ApproachAvoidanceXZ,
                       'Acceleration 3D':Plotter.Acceleration3D,
                       'Distance 3D':Plotter.Trajectory3D
                       }

    Layout = GenLayout(list(PlottingOptions.keys()), SubplotLayout)

    Funcs = [None] * (SubplotLayout[0] * SubplotLayout[1])
    FuncArgs = [None] * (SubplotLayout[0] * SubplotLayout[1])
    PosMap = {}
    counter = 0
    for row in range(SubplotLayout[0]):
        for col in range(SubplotLayout[1]):
            PosMap.update({'{},{}'.format(row, col):counter})
            counter += 1

    window = sg.Window(Title, Layout)

    NewWindow = False
    cancelled = False
    while True:
        event, value = window.read()
        if event in (sg.WIN_CLOSED, '-CANCEL-'):
            cancelled = True
            break
        if event == '-SETTINGS-':
            Settings = SettingWindow()
            if Settings is not None:
                NewWindow = True
                break
        if event == '-SHOW_PLOT-':
            Plotter.MultiPlot(SubplotLayout, Funcs, FuncArgs)
            Plotter.ShowPlots()
        if event.startswith('-PLOT_'):
            ID = event.split('-PLOT_')[-1]
            if value[event] in PlottingOptions.keys() and value[event] != 'None':
                window['-OPTIONS_{}'.format(ID)].update(disabled = False)
            else:
                window['-OPTIONS_{}'.format(ID)].update(disabled = True)
        if event.startswith('-OPTIONS_'):
            ID = event.split('-OPTIONS_')[-1]
            FuncParams = OptionWindow(value['-PLOT_{}'.format(ID)], PlottingOptions, Stimuli, Data)
            NewToolTip = ''
            if FuncParams is not None:
                for key in FuncParams.keys():
                    if key != 'DF':
                        NewToolTip = NewToolTip + "{} = {}\n".format(key, FuncParams[key])
                window['-FRAME_{}'.format(ID)].SetTooltip(NewToolTip)
                ID_Nums = ID.split('_')
                Pos = PosMap['{},{}'.format(int(ID_Nums[0]), int(ID_Nums[-1][0:-1]))]
                FuncArgs[Pos] = FuncParams
                Funcs[Pos] = PlottingOptions[value['-PLOT_{}'.format(ID)]]

    window.close()

    if NewWindow:
        PlotDefinition(Title, Data, Settings['PlotLayout'])

    return None

Title = 'Interactive AAT Plotter'

FilePaths = LoadFromFile(Title)
if FilePaths is not None:
    Data = LoadData(Title, FilePaths['-FILE_PATH-'])
    PlotDefinition(Title, Data, (1, 1))