#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2024.2.5),
    on Апрель 12, 2025, at 20:59
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '3'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2024.2.5'
expName = 'AB'  # from the Builder filename that created this script
# information about this experiment
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'order [1: single-dual or 2: dual-single]': '1',
    'refresh rate (Hz)': '60',
    'port(parallel/serial/cedrus)': 'parallel',
    'port address': '0xC000',
    'language': 'en',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = [1920, 1080]
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='C:\\Users\\Александра Косаченко\\Documents\\GitHub\\Luck1996_presentationCode\\Luck1996AB_lastrun.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('warning')
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log')
    if PILOTING:
        logFile.setLevel(
            prefs.piloting['pilotLoggingLevel']
        )
    else:
        logFile.setLevel(
            logging.getLevel('exp')
        )
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowGUI=False, allowStencil=False,
            monitor='testMonitor', color=[-0.75,-0.75,-0.75], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height',
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [-0.75,-0.75,-0.75]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win._monitorFrameRate = win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate
    win.hideMessage()
    # show a visual indicator if we're in piloting mode
    if PILOTING and prefs.piloting['showPilotingIndicator']:
        win.showPilotingIndicator()
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    # Setup iohub experiment
    ioConfig['Experiment'] = dict(filename=thisExp.dataFileName)
    
    # Start ioHub server
    ioServer = io.launchHubServer(window=win, **ioConfig)
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='iohub'
        )
    if deviceManager.getDevice('key_resp') is None:
        # initialise key_resp
        key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp',
        )
    if deviceManager.getDevice('key_resp_T1') is None:
        # initialise key_resp_T1
        key_resp_T1 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_T1',
        )
    if deviceManager.getDevice('key_resp_T2') is None:
        # initialise key_resp_T2
        key_resp_T2 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_T2',
        )
    if deviceManager.getDevice('press_to_continue') is None:
        # initialise press_to_continue
        press_to_continue = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='press_to_continue',
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='ioHub',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure window is set to foreground to prevent losing focus
    win.winHandle.activate()
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ioHub'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "set_block" ---
    # Run 'Begin Experiment' code from settings_code
    import random
    import pandas as pd
    
    # serial/parallel port address 
    port_address = expInfo['port address']
    port_type = expInfo['port(parallel/serial/cedrus)']
    
    if port_type == 'parallel':
        from psychopy import parallel
        # create parallel port
        p_port = parallel.ParallelPort(address = port_address)
        #send pport trigger
        def send_trigger(triggerCode):
            win.callOnFlip(p_port.setData, triggerCode)
    elif port_type == 'serial':
        # create serial port
        import serial
        s_port = serial.Serial(port = port_address)
        def send_trigger(triggerCode):
            win.callOnFlip(s_port.write, str.encode(str(triggerCode)))
    elif port_type == 'cedrus':
        import pyxid2
        import time
        # get a list of all attached XID devices
        devices = pyxid2.get_xid_devices()
        dev = devices[0] # get the first device to use
        dev.set_pulse_duration(300)
        def send_trigger(triggerCode):
            win.callOnFlip(dev.activate_line, triggerCode)
    
    #local path to language
    local_path = 'languages/'+expInfo['language']
    
    with open(f'{local_path}/rest.txt', 'r') as file:
        txt_rest = file.read()
    
    with open(f'{local_path}/thanks.txt', 'r') as file:
        txt_thanks = file.read()
    
    # duration in frames
    #dur_frames = round(0.083 * int(expInfo['refresh rate (Hz)']))
    dur_frames_stim = round(0.033 * int(expInfo['refresh rate (Hz)']))
    dur_frames_blank = round(0.050 * int(expInfo['refresh rate (Hz)']))
    
    # set trials order
    n = 30 # num of related and unrelated trials
    related_idx = list(range(360))
    unrelated_idx = list(range(360,720))
    random.shuffle(related_idx)
    random.shuffle(unrelated_idx)
    
    # read file with distractors 
    distractors_df = pd.read_csv(f'{local_path}/distractors.csv')
    distractors_df = distractors_df.sample(frac = 1)
    dist_idx = 0
    
    # feedback counters
    counter_T1 = 0
    counter_T2 = 0
    
    # --- Initialize components for Routine "instructions" ---
    text_instructions = visual.TextStim(win=win, name='text_instructions',
        text=None,
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.02, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp = keyboard.Keyboard(deviceName='key_resp')
    
    # --- Initialize components for Routine "set_trial" ---
    
    # --- Initialize components for Routine "T0_context" ---
    context_word = visual.TextStim(win=win, name='context_word',
        text='',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, 1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    blankContext = visual.TextStim(win=win, name='blankContext',
        text=None,
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "distractors_T0" ---
    text_distr_T0 = visual.TextStim(win=win, name='text_distr_T0',
        text='',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='blue', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    text_blank_distractors_T0 = visual.TextStim(win=win, name='text_blank_distractors_T0',
        text=None,
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "T1_number" ---
    text_T1 = visual.TextStim(win=win, name='text_T1',
        text='',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='blue', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    text_blank_number_T1 = visual.TextStim(win=win, name='text_blank_number_T1',
        text=None,
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "distractors_T1" ---
    text_distr_T1 = visual.TextStim(win=win, name='text_distr_T1',
        text='',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='blue', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    text_blank_distractors_T1 = visual.TextStim(win=win, name='text_blank_distractors_T1',
        text=None,
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "T2_probe" ---
    text_T2 = visual.TextStim(win=win, name='text_T2',
        text='',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='red', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    text_blank_T2_probe = visual.TextStim(win=win, name='text_blank_T2_probe',
        text=None,
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "distractors_T2" ---
    text_distr_T2 = visual.TextStim(win=win, name='text_distr_T2',
        text='',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='blue', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    text_blank_distractors_T2 = visual.TextStim(win=win, name='text_blank_distractors_T2',
        text=None,
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "blank_1s" ---
    text_blank1s = visual.TextStim(win=win, name='text_blank1s',
        text=None,
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "Ask_T1" ---
    text_T1ask = visual.TextStim(win=win, name='text_T1ask',
        text='?',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, 1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_T1 = keyboard.Keyboard(deviceName='key_resp_T1')
    
    # --- Initialize components for Routine "blank_T1T2" ---
    text_blankT1T2 = visual.TextStim(win=win, name='text_blankT1T2',
        text=None,
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "Ask_T2" ---
    text_T2ask = visual.TextStim(win=win, name='text_T2ask',
        text='?',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, 1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_T2 = keyboard.Keyboard(deviceName='key_resp_T2')
    
    # --- Initialize components for Routine "blank_2s" ---
    text_blank2s = visual.TextStim(win=win, name='text_blank2s',
        text=None,
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "rest" ---
    text_rest = visual.TextStim(win=win, name='text_rest',
        text='',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    press_to_continue = keyboard.Keyboard(deviceName='press_to_continue')
    
    # --- Initialize components for Routine "end" ---
    text_thanks = visual.TextStim(win=win, name='text_thanks',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # set up handler to look after randomisation of conditions etc
    block = data.TrialHandler2(
        name='block',
        nReps=1.0, 
        method='sequential', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('task_type.xlsx'), 
        seed=None, 
    )
    thisExp.addLoop(block)  # add the loop to the experiment
    thisBlock = block.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisBlock.rgb)
    if thisBlock != None:
        for paramName in thisBlock:
            globals()[paramName] = thisBlock[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisBlock in block:
        currentLoop = block
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisBlock.rgb)
        if thisBlock != None:
            for paramName in thisBlock:
                globals()[paramName] = thisBlock[paramName]
        
        # --- Prepare to start Routine "set_block" ---
        # create an object to store info about Routine set_block
        set_block = data.Routine(
            name='set_block',
            components=[],
        )
        set_block.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from settings_code
        # set blocks order
        task = task1 if expInfo['order [1: single-dual or 2: dual-single]']=='1' else task2
                
        # choose rows for this trial
        trials_in_block = related_idx[-n:] + unrelated_idx[-n:]
        shuffle(trials_in_block)
        related_idx = related_idx[:len(related_idx)-n]
        unrelated_idx = unrelated_idx[:len(unrelated_idx)-n]
        
        # log
        print(f'\nBlock #{block.thisN+1}\nTask: {task}\n')
        # store start times for set_block
        set_block.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        set_block.tStart = globalClock.getTime(format='float')
        set_block.status = STARTED
        thisExp.addData('set_block.started', set_block.tStart)
        set_block.maxDuration = None
        # keep track of which components have finished
        set_blockComponents = set_block.components
        for thisComponent in set_block.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "set_block" ---
        # if trial has changed, end Routine now
        if isinstance(block, data.TrialHandler2) and thisBlock.thisN != block.thisTrial.thisN:
            continueRoutine = False
        set_block.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                set_block.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in set_block.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "set_block" ---
        for thisComponent in set_block.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for set_block
        set_block.tStop = globalClock.getTime(format='float')
        set_block.tStopRefresh = tThisFlipGlobal
        thisExp.addData('set_block.stopped', set_block.tStop)
        # the Routine "set_block" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "instructions" ---
        # create an object to store info about Routine instructions
        instructions = data.Routine(
            name='instructions',
            components=[text_instructions, key_resp],
        )
        instructions.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        text_instructions.setText('')
        # Run 'Begin Routine' code from code_instr
        # open files with instructions
        if block.thisN in [0,6]:
            if task == 'single':
                with open(f'{local_path}/instruction_singleTask.txt', 'r') as file:
                    text_instructions.text = file.read()
            if task == 'dual':
                with open(f'{local_path}/instruction_dualTask.txt', 'r') as file:
                    text_instructions.text = file.read()
        else:
            continueRoutine = False
        # create starting attributes for key_resp
        key_resp.keys = []
        key_resp.rt = []
        _key_resp_allKeys = []
        # store start times for instructions
        instructions.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        instructions.tStart = globalClock.getTime(format='float')
        instructions.status = STARTED
        thisExp.addData('instructions.started', instructions.tStart)
        instructions.maxDuration = None
        # keep track of which components have finished
        instructionsComponents = instructions.components
        for thisComponent in instructions.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "instructions" ---
        # if trial has changed, end Routine now
        if isinstance(block, data.TrialHandler2) and thisBlock.thisN != block.thisTrial.thisN:
            continueRoutine = False
        instructions.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_instructions* updates
            
            # if text_instructions is starting this frame...
            if text_instructions.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_instructions.frameNStart = frameN  # exact frame index
                text_instructions.tStart = t  # local t and not account for scr refresh
                text_instructions.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_instructions, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_instructions.started')
                # update status
                text_instructions.status = STARTED
                text_instructions.setAutoDraw(True)
            
            # if text_instructions is active this frame...
            if text_instructions.status == STARTED:
                # update params
                pass
            
            # *key_resp* updates
            waitOnFlip = False
            
            # if key_resp is starting this frame...
            if key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp.frameNStart = frameN  # exact frame index
                key_resp.tStart = t  # local t and not account for scr refresh
                key_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp.started')
                # update status
                key_resp.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp.status == STARTED and not waitOnFlip:
                theseKeys = key_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_allKeys.extend(theseKeys)
                if len(_key_resp_allKeys):
                    key_resp.keys = _key_resp_allKeys[-1].name  # just the last key pressed
                    key_resp.rt = _key_resp_allKeys[-1].rt
                    key_resp.duration = _key_resp_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                instructions.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in instructions.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "instructions" ---
        for thisComponent in instructions.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for instructions
        instructions.tStop = globalClock.getTime(format='float')
        instructions.tStopRefresh = tThisFlipGlobal
        thisExp.addData('instructions.stopped', instructions.tStop)
        # check responses
        if key_resp.keys in ['', [], None]:  # No response was made
            key_resp.keys = None
        block.addData('key_resp.keys',key_resp.keys)
        if key_resp.keys != None:  # we had a response
            block.addData('key_resp.rt', key_resp.rt)
            block.addData('key_resp.duration', key_resp.duration)
        # the Routine "instructions" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # set up handler to look after randomisation of conditions etc
        trial = data.TrialHandler2(
            name='trial',
            nReps=1.0, 
            method='sequential', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=data.importConditions(
            f'{local_path}/wordPairs.xlsx', 
            selection=trials_in_block
        )
        , 
            seed=None, 
        )
        thisExp.addLoop(trial)  # add the loop to the experiment
        thisTrial = trial.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
        if thisTrial != None:
            for paramName in thisTrial:
                globals()[paramName] = thisTrial[paramName]
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        for thisTrial in trial:
            currentLoop = trial
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
            if thisTrial != None:
                for paramName in thisTrial:
                    globals()[paramName] = thisTrial[paramName]
            
            # --- Prepare to start Routine "set_trial" ---
            # create an object to store info about Routine set_trial
            set_trial = data.Routine(
                name='set_trial',
                components=[],
            )
            set_trial.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from code_trial
            distractors_trial = list(distractors_df.dist[dist_idx:dist_idx+18])
            i = 0 # distractor index
            print(f'\nTrial #{trial.thisN+1}')
            
            #triggers
            if task == 'single':
                trigger_T0 = 10
                trigger_T1 = 20 if responseT1 == 'j' else 21
                trigger_T2 = 40 if related == 'related' else 41
            else:
                trigger_T0 = 11
                trigger_T1 = 22 if responseT1 == 'j' else 23
                trigger_T2 = 42 if related == 'related' else 43
            
            # store start times for set_trial
            set_trial.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            set_trial.tStart = globalClock.getTime(format='float')
            set_trial.status = STARTED
            thisExp.addData('set_trial.started', set_trial.tStart)
            set_trial.maxDuration = None
            # keep track of which components have finished
            set_trialComponents = set_trial.components
            for thisComponent in set_trial.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "set_trial" ---
            # if trial has changed, end Routine now
            if isinstance(trial, data.TrialHandler2) and thisTrial.thisN != trial.thisTrial.thisN:
                continueRoutine = False
            set_trial.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    set_trial.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in set_trial.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "set_trial" ---
            for thisComponent in set_trial.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for set_trial
            set_trial.tStop = globalClock.getTime(format='float')
            set_trial.tStopRefresh = tThisFlipGlobal
            thisExp.addData('set_trial.stopped', set_trial.tStop)
            # Run 'End Routine' code from code_trial
            dist_idx += 18
            # the Routine "set_trial" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "T0_context" ---
            # create an object to store info about Routine T0_context
            T0_context = data.Routine(
                name='T0_context',
                components=[context_word, blankContext],
            )
            T0_context.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            context_word.setText(T0)
            # Run 'Begin Routine' code from code_T0
            stimulus_pulse_started = False
            stimulus_pulse_ended = False
            # store start times for T0_context
            T0_context.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            T0_context.tStart = globalClock.getTime(format='float')
            T0_context.status = STARTED
            thisExp.addData('T0_context.started', T0_context.tStart)
            T0_context.maxDuration = None
            # keep track of which components have finished
            T0_contextComponents = T0_context.components
            for thisComponent in T0_context.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "T0_context" ---
            # if trial has changed, end Routine now
            if isinstance(trial, data.TrialHandler2) and thisTrial.thisN != trial.thisTrial.thisN:
                continueRoutine = False
            T0_context.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 2.0:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *context_word* updates
                
                # if context_word is starting this frame...
                if context_word.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    context_word.frameNStart = frameN  # exact frame index
                    context_word.tStart = t  # local t and not account for scr refresh
                    context_word.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(context_word, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'context_word.started')
                    # update status
                    context_word.status = STARTED
                    context_word.setAutoDraw(True)
                
                # if context_word is active this frame...
                if context_word.status == STARTED:
                    # update params
                    pass
                
                # if context_word is stopping this frame...
                if context_word.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > context_word.tStartRefresh + 1.0-frameTolerance:
                        # keep track of stop time/frame for later
                        context_word.tStop = t  # not accounting for scr refresh
                        context_word.tStopRefresh = tThisFlipGlobal  # on global time
                        context_word.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'context_word.stopped')
                        # update status
                        context_word.status = FINISHED
                        context_word.setAutoDraw(False)
                
                # *blankContext* updates
                
                # if blankContext is starting this frame...
                if blankContext.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                    # keep track of start time/frame for later
                    blankContext.frameNStart = frameN  # exact frame index
                    blankContext.tStart = t  # local t and not account for scr refresh
                    blankContext.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(blankContext, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'blankContext.started')
                    # update status
                    blankContext.status = STARTED
                    blankContext.setAutoDraw(True)
                
                # if blankContext is active this frame...
                if blankContext.status == STARTED:
                    # update params
                    pass
                
                # if blankContext is stopping this frame...
                if blankContext.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > blankContext.tStartRefresh + 1.0-frameTolerance:
                        # keep track of stop time/frame for later
                        blankContext.tStop = t  # not accounting for scr refresh
                        blankContext.tStopRefresh = tThisFlipGlobal  # on global time
                        blankContext.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'blankContext.stopped')
                        # update status
                        blankContext.status = FINISHED
                        blankContext.setAutoDraw(False)
                # Run 'Each Frame' code from code_T0
                #send T0 trigger (10/11)
                if context_word.status == STARTED and not stimulus_pulse_started:
                    stimulus_pulse_started = True
                    stimulus_pulse_start_time = globalClock.getTime()
                    send_trigger(trigger_T0)
                #reset trigger (0)      
                if stimulus_pulse_started and not stimulus_pulse_ended:
                    if globalClock.getTime() - stimulus_pulse_start_time >= 0.005:
                        send_trigger(0)
                        stimulus_pulse_ended = True
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    T0_context.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in T0_context.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "T0_context" ---
            for thisComponent in T0_context.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for T0_context
            T0_context.tStop = globalClock.getTime(format='float')
            T0_context.tStopRefresh = tThisFlipGlobal
            thisExp.addData('T0_context.stopped', T0_context.tStop)
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if T0_context.maxDurationReached:
                routineTimer.addTime(-T0_context.maxDuration)
            elif T0_context.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-2.000000)
            
            # set up handler to look after randomisation of conditions etc
            trials_T0_context = data.TrialHandler2(
                name='trials_T0_context',
                nReps=10.0, 
                method='sequential', 
                extraInfo=expInfo, 
                originPath=-1, 
                trialList=[None], 
                seed=None, 
            )
            thisExp.addLoop(trials_T0_context)  # add the loop to the experiment
            thisTrials_T0_context = trials_T0_context.trialList[0]  # so we can initialise stimuli with some values
            # abbreviate parameter names if possible (e.g. rgb = thisTrials_T0_context.rgb)
            if thisTrials_T0_context != None:
                for paramName in thisTrials_T0_context:
                    globals()[paramName] = thisTrials_T0_context[paramName]
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            
            for thisTrials_T0_context in trials_T0_context:
                currentLoop = trials_T0_context
                thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
                if thisSession is not None:
                    # if running in a Session with a Liaison client, send data up to now
                    thisSession.sendExperimentData()
                # abbreviate parameter names if possible (e.g. rgb = thisTrials_T0_context.rgb)
                if thisTrials_T0_context != None:
                    for paramName in thisTrials_T0_context:
                        globals()[paramName] = thisTrials_T0_context[paramName]
                
                # --- Prepare to start Routine "distractors_T0" ---
                # create an object to store info about Routine distractors_T0
                distractors_T0 = data.Routine(
                    name='distractors_T0',
                    components=[text_distr_T0, text_blank_distractors_T0],
                )
                distractors_T0.status = NOT_STARTED
                continueRoutine = True
                # update component parameters for each repeat
                # Run 'Begin Routine' code from breakLoop_T0
                thisExp.addData('distractor_T0',distractors_trial[i])
                stimulus_pulse_started = False
                stimulus_pulse_ended = False
                text_distr_T0.setText(distractors_trial[i])
                # store start times for distractors_T0
                distractors_T0.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
                distractors_T0.tStart = globalClock.getTime(format='float')
                distractors_T0.status = STARTED
                thisExp.addData('distractors_T0.started', distractors_T0.tStart)
                distractors_T0.maxDuration = None
                # keep track of which components have finished
                distractors_T0Components = distractors_T0.components
                for thisComponent in distractors_T0.components:
                    thisComponent.tStart = None
                    thisComponent.tStop = None
                    thisComponent.tStartRefresh = None
                    thisComponent.tStopRefresh = None
                    if hasattr(thisComponent, 'status'):
                        thisComponent.status = NOT_STARTED
                # reset timers
                t = 0
                _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                frameN = -1
                
                # --- Run Routine "distractors_T0" ---
                # if trial has changed, end Routine now
                if isinstance(trials_T0_context, data.TrialHandler2) and thisTrials_T0_context.thisN != trials_T0_context.thisTrial.thisN:
                    continueRoutine = False
                distractors_T0.forceEnded = routineForceEnded = not continueRoutine
                while continueRoutine:
                    # get current time
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                    # update/draw components on each frame
                    # Run 'Each Frame' code from breakLoop_T0
                    #send dist_T0 trigger (30)
                    if text_distr_T0.status == STARTED and not stimulus_pulse_started:
                        stimulus_pulse_started = True
                        stimulus_pulse_start_time = globalClock.getTime()
                        send_trigger(30)
                    #reset trigger (0)      
                    if stimulus_pulse_started and not stimulus_pulse_ended:
                        if globalClock.getTime() - stimulus_pulse_start_time >= 0.005:
                            send_trigger(0)
                            stimulus_pulse_ended = True
                    
                    # *text_distr_T0* updates
                    
                    # if text_distr_T0 is starting this frame...
                    if text_distr_T0.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                        # keep track of start time/frame for later
                        text_distr_T0.frameNStart = frameN  # exact frame index
                        text_distr_T0.tStart = t  # local t and not account for scr refresh
                        text_distr_T0.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(text_distr_T0, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'text_distr_T0.started')
                        # update status
                        text_distr_T0.status = STARTED
                        text_distr_T0.setAutoDraw(True)
                    
                    # if text_distr_T0 is active this frame...
                    if text_distr_T0.status == STARTED:
                        # update params
                        pass
                    
                    # if text_distr_T0 is stopping this frame...
                    if text_distr_T0.status == STARTED:
                        if frameN >= (text_distr_T0.frameNStart + dur_frames_stim):
                            # keep track of stop time/frame for later
                            text_distr_T0.tStop = t  # not accounting for scr refresh
                            text_distr_T0.tStopRefresh = tThisFlipGlobal  # on global time
                            text_distr_T0.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'text_distr_T0.stopped')
                            # update status
                            text_distr_T0.status = FINISHED
                            text_distr_T0.setAutoDraw(False)
                    
                    # *text_blank_distractors_T0* updates
                    
                    # if text_blank_distractors_T0 is starting this frame...
                    if text_blank_distractors_T0.status == NOT_STARTED and frameN >= dur_frames_stim:
                        # keep track of start time/frame for later
                        text_blank_distractors_T0.frameNStart = frameN  # exact frame index
                        text_blank_distractors_T0.tStart = t  # local t and not account for scr refresh
                        text_blank_distractors_T0.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(text_blank_distractors_T0, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'text_blank_distractors_T0.started')
                        # update status
                        text_blank_distractors_T0.status = STARTED
                        text_blank_distractors_T0.setAutoDraw(True)
                    
                    # if text_blank_distractors_T0 is active this frame...
                    if text_blank_distractors_T0.status == STARTED:
                        # update params
                        pass
                    
                    # if text_blank_distractors_T0 is stopping this frame...
                    if text_blank_distractors_T0.status == STARTED:
                        if frameN >= (text_blank_distractors_T0.frameNStart + dur_frames_blank):
                            # keep track of stop time/frame for later
                            text_blank_distractors_T0.tStop = t  # not accounting for scr refresh
                            text_blank_distractors_T0.tStopRefresh = tThisFlipGlobal  # on global time
                            text_blank_distractors_T0.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'text_blank_distractors_T0.stopped')
                            # update status
                            text_blank_distractors_T0.status = FINISHED
                            text_blank_distractors_T0.setAutoDraw(False)
                    
                    # check for quit (typically the Esc key)
                    if defaultKeyboard.getKeys(keyList=["escape"]):
                        thisExp.status = FINISHED
                    if thisExp.status == FINISHED or endExpNow:
                        endExperiment(thisExp, win=win)
                        return
                    # pause experiment here if requested
                    if thisExp.status == PAUSED:
                        pauseExperiment(
                            thisExp=thisExp, 
                            win=win, 
                            timers=[routineTimer], 
                            playbackComponents=[]
                        )
                        # skip the frame we paused on
                        continue
                    
                    # check if all components have finished
                    if not continueRoutine:  # a component has requested a forced-end of Routine
                        distractors_T0.forceEnded = routineForceEnded = True
                        break
                    continueRoutine = False  # will revert to True if at least one component still running
                    for thisComponent in distractors_T0.components:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished
                    
                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()
                
                # --- Ending Routine "distractors_T0" ---
                for thisComponent in distractors_T0.components:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                # store stop times for distractors_T0
                distractors_T0.tStop = globalClock.getTime(format='float')
                distractors_T0.tStopRefresh = tThisFlipGlobal
                thisExp.addData('distractors_T0.stopped', distractors_T0.tStop)
                # Run 'End Routine' code from breakLoop_T0
                i += 1
                if trials_T0_context.thisN == lagT0-2:
                    trials_T0_context.finished = True
                
                # the Routine "distractors_T0" was not non-slip safe, so reset the non-slip timer
                routineTimer.reset()
                thisExp.nextEntry()
                
            # completed 10.0 repeats of 'trials_T0_context'
            
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            
            # --- Prepare to start Routine "T1_number" ---
            # create an object to store info about Routine T1_number
            T1_number = data.Routine(
                name='T1_number',
                components=[text_T1, text_blank_number_T1],
            )
            T1_number.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            text_T1.setText(str(int(T1)))
            # Run 'Begin Routine' code from code_T1
            stimulus_pulse_started = False
            stimulus_pulse_ended = False
            # store start times for T1_number
            T1_number.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            T1_number.tStart = globalClock.getTime(format='float')
            T1_number.status = STARTED
            thisExp.addData('T1_number.started', T1_number.tStart)
            T1_number.maxDuration = None
            # keep track of which components have finished
            T1_numberComponents = T1_number.components
            for thisComponent in T1_number.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "T1_number" ---
            # if trial has changed, end Routine now
            if isinstance(trial, data.TrialHandler2) and thisTrial.thisN != trial.thisTrial.thisN:
                continueRoutine = False
            T1_number.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *text_T1* updates
                
                # if text_T1 is starting this frame...
                if text_T1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text_T1.frameNStart = frameN  # exact frame index
                    text_T1.tStart = t  # local t and not account for scr refresh
                    text_T1.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_T1, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_T1.started')
                    # update status
                    text_T1.status = STARTED
                    text_T1.setAutoDraw(True)
                
                # if text_T1 is active this frame...
                if text_T1.status == STARTED:
                    # update params
                    pass
                
                # if text_T1 is stopping this frame...
                if text_T1.status == STARTED:
                    if frameN >= (text_T1.frameNStart + dur_frames_stim):
                        # keep track of stop time/frame for later
                        text_T1.tStop = t  # not accounting for scr refresh
                        text_T1.tStopRefresh = tThisFlipGlobal  # on global time
                        text_T1.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'text_T1.stopped')
                        # update status
                        text_T1.status = FINISHED
                        text_T1.setAutoDraw(False)
                
                # *text_blank_number_T1* updates
                
                # if text_blank_number_T1 is starting this frame...
                if text_blank_number_T1.status == NOT_STARTED and frameN >= dur_frames_stim:
                    # keep track of start time/frame for later
                    text_blank_number_T1.frameNStart = frameN  # exact frame index
                    text_blank_number_T1.tStart = t  # local t and not account for scr refresh
                    text_blank_number_T1.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_blank_number_T1, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_blank_number_T1.started')
                    # update status
                    text_blank_number_T1.status = STARTED
                    text_blank_number_T1.setAutoDraw(True)
                
                # if text_blank_number_T1 is active this frame...
                if text_blank_number_T1.status == STARTED:
                    # update params
                    pass
                
                # if text_blank_number_T1 is stopping this frame...
                if text_blank_number_T1.status == STARTED:
                    if frameN >= (text_blank_number_T1.frameNStart + dur_frames_blank):
                        # keep track of stop time/frame for later
                        text_blank_number_T1.tStop = t  # not accounting for scr refresh
                        text_blank_number_T1.tStopRefresh = tThisFlipGlobal  # on global time
                        text_blank_number_T1.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'text_blank_number_T1.stopped')
                        # update status
                        text_blank_number_T1.status = FINISHED
                        text_blank_number_T1.setAutoDraw(False)
                # Run 'Each Frame' code from code_T1
                #send T1 trigger (20/21/22/23)
                if text_T1.status == STARTED and not stimulus_pulse_started:
                    stimulus_pulse_started = True
                    stimulus_pulse_start_time = globalClock.getTime()
                    send_trigger(trigger_T1)
                #reset trigger (0)      
                if stimulus_pulse_started and not stimulus_pulse_ended:
                    if globalClock.getTime() - stimulus_pulse_start_time >= 0.005:
                        send_trigger(0)
                        stimulus_pulse_ended = True
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    T1_number.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in T1_number.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "T1_number" ---
            for thisComponent in T1_number.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for T1_number
            T1_number.tStop = globalClock.getTime(format='float')
            T1_number.tStopRefresh = tThisFlipGlobal
            thisExp.addData('T1_number.stopped', T1_number.tStop)
            # the Routine "T1_number" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # set up handler to look after randomisation of conditions etc
            trials_T1 = data.TrialHandler2(
                name='trials_T1',
                nReps=7.0, 
                method='sequential', 
                extraInfo=expInfo, 
                originPath=-1, 
                trialList=[None], 
                seed=None, 
            )
            thisExp.addLoop(trials_T1)  # add the loop to the experiment
            thisTrials_T1 = trials_T1.trialList[0]  # so we can initialise stimuli with some values
            # abbreviate parameter names if possible (e.g. rgb = thisTrials_T1.rgb)
            if thisTrials_T1 != None:
                for paramName in thisTrials_T1:
                    globals()[paramName] = thisTrials_T1[paramName]
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            
            for thisTrials_T1 in trials_T1:
                currentLoop = trials_T1
                thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
                if thisSession is not None:
                    # if running in a Session with a Liaison client, send data up to now
                    thisSession.sendExperimentData()
                # abbreviate parameter names if possible (e.g. rgb = thisTrials_T1.rgb)
                if thisTrials_T1 != None:
                    for paramName in thisTrials_T1:
                        globals()[paramName] = thisTrials_T1[paramName]
                
                # --- Prepare to start Routine "distractors_T1" ---
                # create an object to store info about Routine distractors_T1
                distractors_T1 = data.Routine(
                    name='distractors_T1',
                    components=[text_distr_T1, text_blank_distractors_T1],
                )
                distractors_T1.status = NOT_STARTED
                continueRoutine = True
                # update component parameters for each repeat
                # Run 'Begin Routine' code from skipLoop_ifLag1
                if lagT1 == 1:
                     continueRoutine = False
                     thisExp.addData('distractor_T1','')
                else:
                    i += 1
                    thisExp.addData('distractor_T1',distractors_trial[i])
                    stimulus_pulse_started = False
                    stimulus_pulse_ended = False
                text_distr_T1.setText(distractors_trial[i])
                # store start times for distractors_T1
                distractors_T1.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
                distractors_T1.tStart = globalClock.getTime(format='float')
                distractors_T1.status = STARTED
                thisExp.addData('distractors_T1.started', distractors_T1.tStart)
                distractors_T1.maxDuration = None
                # keep track of which components have finished
                distractors_T1Components = distractors_T1.components
                for thisComponent in distractors_T1.components:
                    thisComponent.tStart = None
                    thisComponent.tStop = None
                    thisComponent.tStartRefresh = None
                    thisComponent.tStopRefresh = None
                    if hasattr(thisComponent, 'status'):
                        thisComponent.status = NOT_STARTED
                # reset timers
                t = 0
                _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                frameN = -1
                
                # --- Run Routine "distractors_T1" ---
                # if trial has changed, end Routine now
                if isinstance(trials_T1, data.TrialHandler2) and thisTrials_T1.thisN != trials_T1.thisTrial.thisN:
                    continueRoutine = False
                distractors_T1.forceEnded = routineForceEnded = not continueRoutine
                while continueRoutine:
                    # get current time
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                    # update/draw components on each frame
                    # Run 'Each Frame' code from skipLoop_ifLag1
                    if lagT1 != 1:
                        #send dist_T1 trigger (30)
                        if text_distr_T1.status == STARTED and not stimulus_pulse_started:
                            stimulus_pulse_started = True
                            stimulus_pulse_start_time = globalClock.getTime()
                            send_trigger(30)
                        #reset trigger (0)      
                        if stimulus_pulse_started and not stimulus_pulse_ended:
                            if globalClock.getTime() - stimulus_pulse_start_time >= 0.005:
                                send_trigger(0)
                                stimulus_pulse_ended = True
                    
                    # *text_distr_T1* updates
                    
                    # if text_distr_T1 is starting this frame...
                    if text_distr_T1.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                        # keep track of start time/frame for later
                        text_distr_T1.frameNStart = frameN  # exact frame index
                        text_distr_T1.tStart = t  # local t and not account for scr refresh
                        text_distr_T1.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(text_distr_T1, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'text_distr_T1.started')
                        # update status
                        text_distr_T1.status = STARTED
                        text_distr_T1.setAutoDraw(True)
                    
                    # if text_distr_T1 is active this frame...
                    if text_distr_T1.status == STARTED:
                        # update params
                        pass
                    
                    # if text_distr_T1 is stopping this frame...
                    if text_distr_T1.status == STARTED:
                        if frameN >= (text_distr_T1.frameNStart + dur_frames_stim):
                            # keep track of stop time/frame for later
                            text_distr_T1.tStop = t  # not accounting for scr refresh
                            text_distr_T1.tStopRefresh = tThisFlipGlobal  # on global time
                            text_distr_T1.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'text_distr_T1.stopped')
                            # update status
                            text_distr_T1.status = FINISHED
                            text_distr_T1.setAutoDraw(False)
                    
                    # *text_blank_distractors_T1* updates
                    
                    # if text_blank_distractors_T1 is starting this frame...
                    if text_blank_distractors_T1.status == NOT_STARTED and frameN >= dur_frames_stim:
                        # keep track of start time/frame for later
                        text_blank_distractors_T1.frameNStart = frameN  # exact frame index
                        text_blank_distractors_T1.tStart = t  # local t and not account for scr refresh
                        text_blank_distractors_T1.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(text_blank_distractors_T1, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'text_blank_distractors_T1.started')
                        # update status
                        text_blank_distractors_T1.status = STARTED
                        text_blank_distractors_T1.setAutoDraw(True)
                    
                    # if text_blank_distractors_T1 is active this frame...
                    if text_blank_distractors_T1.status == STARTED:
                        # update params
                        pass
                    
                    # if text_blank_distractors_T1 is stopping this frame...
                    if text_blank_distractors_T1.status == STARTED:
                        if frameN >= (text_blank_distractors_T1.frameNStart + dur_frames_blank):
                            # keep track of stop time/frame for later
                            text_blank_distractors_T1.tStop = t  # not accounting for scr refresh
                            text_blank_distractors_T1.tStopRefresh = tThisFlipGlobal  # on global time
                            text_blank_distractors_T1.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'text_blank_distractors_T1.stopped')
                            # update status
                            text_blank_distractors_T1.status = FINISHED
                            text_blank_distractors_T1.setAutoDraw(False)
                    
                    # check for quit (typically the Esc key)
                    if defaultKeyboard.getKeys(keyList=["escape"]):
                        thisExp.status = FINISHED
                    if thisExp.status == FINISHED or endExpNow:
                        endExperiment(thisExp, win=win)
                        return
                    # pause experiment here if requested
                    if thisExp.status == PAUSED:
                        pauseExperiment(
                            thisExp=thisExp, 
                            win=win, 
                            timers=[routineTimer], 
                            playbackComponents=[]
                        )
                        # skip the frame we paused on
                        continue
                    
                    # check if all components have finished
                    if not continueRoutine:  # a component has requested a forced-end of Routine
                        distractors_T1.forceEnded = routineForceEnded = True
                        break
                    continueRoutine = False  # will revert to True if at least one component still running
                    for thisComponent in distractors_T1.components:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished
                    
                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()
                
                # --- Ending Routine "distractors_T1" ---
                for thisComponent in distractors_T1.components:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                # store stop times for distractors_T1
                distractors_T1.tStop = globalClock.getTime(format='float')
                distractors_T1.tStopRefresh = tThisFlipGlobal
                thisExp.addData('distractors_T1.stopped', distractors_T1.tStop)
                # Run 'End Routine' code from skipLoop_ifLag1
                if lagT1 == 1 or trials_T1.thisN == lagT1-2:
                    trials_T1.finished = True
                # the Routine "distractors_T1" was not non-slip safe, so reset the non-slip timer
                routineTimer.reset()
                thisExp.nextEntry()
                
            # completed 7.0 repeats of 'trials_T1'
            
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            
            # --- Prepare to start Routine "T2_probe" ---
            # create an object to store info about Routine T2_probe
            T2_probe = data.Routine(
                name='T2_probe',
                components=[text_T2, text_blank_T2_probe],
            )
            T2_probe.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            text_T2.setText(T2)
            # Run 'Begin Routine' code from code_T2
            stimulus_pulse_started = False
            stimulus_pulse_ended = False
            # store start times for T2_probe
            T2_probe.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            T2_probe.tStart = globalClock.getTime(format='float')
            T2_probe.status = STARTED
            thisExp.addData('T2_probe.started', T2_probe.tStart)
            T2_probe.maxDuration = None
            # keep track of which components have finished
            T2_probeComponents = T2_probe.components
            for thisComponent in T2_probe.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "T2_probe" ---
            # if trial has changed, end Routine now
            if isinstance(trial, data.TrialHandler2) and thisTrial.thisN != trial.thisTrial.thisN:
                continueRoutine = False
            T2_probe.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *text_T2* updates
                
                # if text_T2 is starting this frame...
                if text_T2.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                    # keep track of start time/frame for later
                    text_T2.frameNStart = frameN  # exact frame index
                    text_T2.tStart = t  # local t and not account for scr refresh
                    text_T2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_T2, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_T2.started')
                    # update status
                    text_T2.status = STARTED
                    text_T2.setAutoDraw(True)
                
                # if text_T2 is active this frame...
                if text_T2.status == STARTED:
                    # update params
                    pass
                
                # if text_T2 is stopping this frame...
                if text_T2.status == STARTED:
                    if frameN >= (text_T2.frameNStart + dur_frames_stim):
                        # keep track of stop time/frame for later
                        text_T2.tStop = t  # not accounting for scr refresh
                        text_T2.tStopRefresh = tThisFlipGlobal  # on global time
                        text_T2.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'text_T2.stopped')
                        # update status
                        text_T2.status = FINISHED
                        text_T2.setAutoDraw(False)
                
                # *text_blank_T2_probe* updates
                
                # if text_blank_T2_probe is starting this frame...
                if text_blank_T2_probe.status == NOT_STARTED and frameN >= dur_frames_stim:
                    # keep track of start time/frame for later
                    text_blank_T2_probe.frameNStart = frameN  # exact frame index
                    text_blank_T2_probe.tStart = t  # local t and not account for scr refresh
                    text_blank_T2_probe.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_blank_T2_probe, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_blank_T2_probe.started')
                    # update status
                    text_blank_T2_probe.status = STARTED
                    text_blank_T2_probe.setAutoDraw(True)
                
                # if text_blank_T2_probe is active this frame...
                if text_blank_T2_probe.status == STARTED:
                    # update params
                    pass
                
                # if text_blank_T2_probe is stopping this frame...
                if text_blank_T2_probe.status == STARTED:
                    if frameN >= (text_blank_T2_probe.frameNStart + dur_frames_blank):
                        # keep track of stop time/frame for later
                        text_blank_T2_probe.tStop = t  # not accounting for scr refresh
                        text_blank_T2_probe.tStopRefresh = tThisFlipGlobal  # on global time
                        text_blank_T2_probe.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'text_blank_T2_probe.stopped')
                        # update status
                        text_blank_T2_probe.status = FINISHED
                        text_blank_T2_probe.setAutoDraw(False)
                # Run 'Each Frame' code from code_T2
                #send T2 trigger (40/41/42/43)
                if text_T2.status == STARTED and not stimulus_pulse_started:
                    stimulus_pulse_started = True
                    stimulus_pulse_start_time = globalClock.getTime()
                    send_trigger(trigger_T2)
                #reset trigger (0)      
                if stimulus_pulse_started and not stimulus_pulse_ended:
                    if globalClock.getTime() - stimulus_pulse_start_time >= 0.005:
                        send_trigger(0)
                        stimulus_pulse_ended = True
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    T2_probe.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in T2_probe.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "T2_probe" ---
            for thisComponent in T2_probe.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for T2_probe
            T2_probe.tStop = globalClock.getTime(format='float')
            T2_probe.tStopRefresh = tThisFlipGlobal
            thisExp.addData('T2_probe.stopped', T2_probe.tStop)
            # the Routine "T2_probe" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # set up handler to look after randomisation of conditions etc
            trials_T2 = data.TrialHandler2(
                name='trials_T2',
                nReps=12.0, 
                method='sequential', 
                extraInfo=expInfo, 
                originPath=-1, 
                trialList=[None], 
                seed=None, 
            )
            thisExp.addLoop(trials_T2)  # add the loop to the experiment
            thisTrials_T2 = trials_T2.trialList[0]  # so we can initialise stimuli with some values
            # abbreviate parameter names if possible (e.g. rgb = thisTrials_T2.rgb)
            if thisTrials_T2 != None:
                for paramName in thisTrials_T2:
                    globals()[paramName] = thisTrials_T2[paramName]
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            
            for thisTrials_T2 in trials_T2:
                currentLoop = trials_T2
                thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
                if thisSession is not None:
                    # if running in a Session with a Liaison client, send data up to now
                    thisSession.sendExperimentData()
                # abbreviate parameter names if possible (e.g. rgb = thisTrials_T2.rgb)
                if thisTrials_T2 != None:
                    for paramName in thisTrials_T2:
                        globals()[paramName] = thisTrials_T2[paramName]
                
                # --- Prepare to start Routine "distractors_T2" ---
                # create an object to store info about Routine distractors_T2
                distractors_T2 = data.Routine(
                    name='distractors_T2',
                    components=[text_distr_T2, text_blank_distractors_T2],
                )
                distractors_T2.status = NOT_STARTED
                continueRoutine = True
                # update component parameters for each repeat
                # Run 'Begin Routine' code from breakLoop_T2
                thisExp.addData('distractor_T2',distractors_trial[i])
                stimulus_pulse_started = False
                stimulus_pulse_ended = False
                text_distr_T2.setText(distractors_trial[i])
                # store start times for distractors_T2
                distractors_T2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
                distractors_T2.tStart = globalClock.getTime(format='float')
                distractors_T2.status = STARTED
                thisExp.addData('distractors_T2.started', distractors_T2.tStart)
                distractors_T2.maxDuration = None
                # keep track of which components have finished
                distractors_T2Components = distractors_T2.components
                for thisComponent in distractors_T2.components:
                    thisComponent.tStart = None
                    thisComponent.tStop = None
                    thisComponent.tStartRefresh = None
                    thisComponent.tStopRefresh = None
                    if hasattr(thisComponent, 'status'):
                        thisComponent.status = NOT_STARTED
                # reset timers
                t = 0
                _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                frameN = -1
                
                # --- Run Routine "distractors_T2" ---
                # if trial has changed, end Routine now
                if isinstance(trials_T2, data.TrialHandler2) and thisTrials_T2.thisN != trials_T2.thisTrial.thisN:
                    continueRoutine = False
                distractors_T2.forceEnded = routineForceEnded = not continueRoutine
                while continueRoutine:
                    # get current time
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                    # update/draw components on each frame
                    # Run 'Each Frame' code from breakLoop_T2
                    #send dist_T2 trigger (30)
                    if text_distr_T2.status == STARTED and not stimulus_pulse_started:
                        stimulus_pulse_started = True
                        stimulus_pulse_start_time = globalClock.getTime()
                        send_trigger(30)
                    #reset trigger (0)      
                    if stimulus_pulse_started and not stimulus_pulse_ended:
                        if globalClock.getTime() - stimulus_pulse_start_time >= 0.005:
                            send_trigger(0)
                            stimulus_pulse_ended = True
                    
                    # *text_distr_T2* updates
                    
                    # if text_distr_T2 is starting this frame...
                    if text_distr_T2.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                        # keep track of start time/frame for later
                        text_distr_T2.frameNStart = frameN  # exact frame index
                        text_distr_T2.tStart = t  # local t and not account for scr refresh
                        text_distr_T2.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(text_distr_T2, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'text_distr_T2.started')
                        # update status
                        text_distr_T2.status = STARTED
                        text_distr_T2.setAutoDraw(True)
                    
                    # if text_distr_T2 is active this frame...
                    if text_distr_T2.status == STARTED:
                        # update params
                        pass
                    
                    # if text_distr_T2 is stopping this frame...
                    if text_distr_T2.status == STARTED:
                        if frameN >= (text_distr_T2.frameNStart + dur_frames_stim):
                            # keep track of stop time/frame for later
                            text_distr_T2.tStop = t  # not accounting for scr refresh
                            text_distr_T2.tStopRefresh = tThisFlipGlobal  # on global time
                            text_distr_T2.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'text_distr_T2.stopped')
                            # update status
                            text_distr_T2.status = FINISHED
                            text_distr_T2.setAutoDraw(False)
                    
                    # *text_blank_distractors_T2* updates
                    
                    # if text_blank_distractors_T2 is starting this frame...
                    if text_blank_distractors_T2.status == NOT_STARTED and frameN >= dur_frames_stim:
                        # keep track of start time/frame for later
                        text_blank_distractors_T2.frameNStart = frameN  # exact frame index
                        text_blank_distractors_T2.tStart = t  # local t and not account for scr refresh
                        text_blank_distractors_T2.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(text_blank_distractors_T2, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'text_blank_distractors_T2.started')
                        # update status
                        text_blank_distractors_T2.status = STARTED
                        text_blank_distractors_T2.setAutoDraw(True)
                    
                    # if text_blank_distractors_T2 is active this frame...
                    if text_blank_distractors_T2.status == STARTED:
                        # update params
                        pass
                    
                    # if text_blank_distractors_T2 is stopping this frame...
                    if text_blank_distractors_T2.status == STARTED:
                        if frameN >= (text_blank_distractors_T2.frameNStart + dur_frames_blank):
                            # keep track of stop time/frame for later
                            text_blank_distractors_T2.tStop = t  # not accounting for scr refresh
                            text_blank_distractors_T2.tStopRefresh = tThisFlipGlobal  # on global time
                            text_blank_distractors_T2.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'text_blank_distractors_T2.stopped')
                            # update status
                            text_blank_distractors_T2.status = FINISHED
                            text_blank_distractors_T2.setAutoDraw(False)
                    
                    # check for quit (typically the Esc key)
                    if defaultKeyboard.getKeys(keyList=["escape"]):
                        thisExp.status = FINISHED
                    if thisExp.status == FINISHED or endExpNow:
                        endExperiment(thisExp, win=win)
                        return
                    # pause experiment here if requested
                    if thisExp.status == PAUSED:
                        pauseExperiment(
                            thisExp=thisExp, 
                            win=win, 
                            timers=[routineTimer], 
                            playbackComponents=[]
                        )
                        # skip the frame we paused on
                        continue
                    
                    # check if all components have finished
                    if not continueRoutine:  # a component has requested a forced-end of Routine
                        distractors_T2.forceEnded = routineForceEnded = True
                        break
                    continueRoutine = False  # will revert to True if at least one component still running
                    for thisComponent in distractors_T2.components:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished
                    
                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()
                
                # --- Ending Routine "distractors_T2" ---
                for thisComponent in distractors_T2.components:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                # store stop times for distractors_T2
                distractors_T2.tStop = globalClock.getTime(format='float')
                distractors_T2.tStopRefresh = tThisFlipGlobal
                thisExp.addData('distractors_T2.stopped', distractors_T2.tStop)
                # Run 'End Routine' code from breakLoop_T2
                i += 1
                if trials_T2.thisN == lagT2-1:
                    trials_T2.finished = True
                
                # the Routine "distractors_T2" was not non-slip safe, so reset the non-slip timer
                routineTimer.reset()
                thisExp.nextEntry()
                
            # completed 12.0 repeats of 'trials_T2'
            
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            
            # --- Prepare to start Routine "blank_1s" ---
            # create an object to store info about Routine blank_1s
            blank_1s = data.Routine(
                name='blank_1s',
                components=[text_blank1s],
            )
            blank_1s.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # store start times for blank_1s
            blank_1s.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            blank_1s.tStart = globalClock.getTime(format='float')
            blank_1s.status = STARTED
            thisExp.addData('blank_1s.started', blank_1s.tStart)
            blank_1s.maxDuration = None
            # keep track of which components have finished
            blank_1sComponents = blank_1s.components
            for thisComponent in blank_1s.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "blank_1s" ---
            # if trial has changed, end Routine now
            if isinstance(trial, data.TrialHandler2) and thisTrial.thisN != trial.thisTrial.thisN:
                continueRoutine = False
            blank_1s.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 1.0:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *text_blank1s* updates
                
                # if text_blank1s is starting this frame...
                if text_blank1s.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text_blank1s.frameNStart = frameN  # exact frame index
                    text_blank1s.tStart = t  # local t and not account for scr refresh
                    text_blank1s.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_blank1s, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_blank1s.started')
                    # update status
                    text_blank1s.status = STARTED
                    text_blank1s.setAutoDraw(True)
                
                # if text_blank1s is active this frame...
                if text_blank1s.status == STARTED:
                    # update params
                    pass
                
                # if text_blank1s is stopping this frame...
                if text_blank1s.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > text_blank1s.tStartRefresh + 1.0-frameTolerance:
                        # keep track of stop time/frame for later
                        text_blank1s.tStop = t  # not accounting for scr refresh
                        text_blank1s.tStopRefresh = tThisFlipGlobal  # on global time
                        text_blank1s.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'text_blank1s.stopped')
                        # update status
                        text_blank1s.status = FINISHED
                        text_blank1s.setAutoDraw(False)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    blank_1s.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in blank_1s.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "blank_1s" ---
            for thisComponent in blank_1s.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for blank_1s
            blank_1s.tStop = globalClock.getTime(format='float')
            blank_1s.tStopRefresh = tThisFlipGlobal
            thisExp.addData('blank_1s.stopped', blank_1s.tStop)
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if blank_1s.maxDurationReached:
                routineTimer.addTime(-blank_1s.maxDuration)
            elif blank_1s.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-1.000000)
            
            # --- Prepare to start Routine "Ask_T1" ---
            # create an object to store info about Routine Ask_T1
            Ask_T1 = data.Routine(
                name='Ask_T1',
                components=[text_T1ask, key_resp_T1],
            )
            Ask_T1.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # create starting attributes for key_resp_T1
            key_resp_T1.keys = []
            key_resp_T1.rt = []
            _key_resp_T1_allKeys = []
            # Run 'Begin Routine' code from code_text_T1ask
            if task == 'dual':
                pressed = False
                stimulus_pulse_started = False
                stimulus_pulse_ended = False
            # store start times for Ask_T1
            Ask_T1.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            Ask_T1.tStart = globalClock.getTime(format='float')
            Ask_T1.status = STARTED
            thisExp.addData('Ask_T1.started', Ask_T1.tStart)
            Ask_T1.maxDuration = None
            # skip Routine Ask_T1 if its 'Skip if' condition is True
            Ask_T1.skipped = continueRoutine and not (task == 'single')
            continueRoutine = Ask_T1.skipped
            # keep track of which components have finished
            Ask_T1Components = Ask_T1.components
            for thisComponent in Ask_T1.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "Ask_T1" ---
            # if trial has changed, end Routine now
            if isinstance(trial, data.TrialHandler2) and thisTrial.thisN != trial.thisTrial.thisN:
                continueRoutine = False
            Ask_T1.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 2.0:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *text_T1ask* updates
                
                # if text_T1ask is starting this frame...
                if text_T1ask.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                    # keep track of start time/frame for later
                    text_T1ask.frameNStart = frameN  # exact frame index
                    text_T1ask.tStart = t  # local t and not account for scr refresh
                    text_T1ask.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_T1ask, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_T1ask.started')
                    # update status
                    text_T1ask.status = STARTED
                    text_T1ask.setAutoDraw(True)
                
                # if text_T1ask is active this frame...
                if text_T1ask.status == STARTED:
                    # update params
                    pass
                
                # if text_T1ask is stopping this frame...
                if text_T1ask.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > text_T1ask.tStartRefresh + 2-frameTolerance:
                        # keep track of stop time/frame for later
                        text_T1ask.tStop = t  # not accounting for scr refresh
                        text_T1ask.tStopRefresh = tThisFlipGlobal  # on global time
                        text_T1ask.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'text_T1ask.stopped')
                        # update status
                        text_T1ask.status = FINISHED
                        text_T1ask.setAutoDraw(False)
                
                # *key_resp_T1* updates
                waitOnFlip = False
                
                # if key_resp_T1 is starting this frame...
                if key_resp_T1.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                    # keep track of start time/frame for later
                    key_resp_T1.frameNStart = frameN  # exact frame index
                    key_resp_T1.tStart = t  # local t and not account for scr refresh
                    key_resp_T1.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(key_resp_T1, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'key_resp_T1.started')
                    # update status
                    key_resp_T1.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(key_resp_T1.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(key_resp_T1.clearEvents, eventType='keyboard')  # clear events on next screen flip
                
                # if key_resp_T1 is stopping this frame...
                if key_resp_T1.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > key_resp_T1.tStartRefresh + 2-frameTolerance:
                        # keep track of stop time/frame for later
                        key_resp_T1.tStop = t  # not accounting for scr refresh
                        key_resp_T1.tStopRefresh = tThisFlipGlobal  # on global time
                        key_resp_T1.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'key_resp_T1.stopped')
                        # update status
                        key_resp_T1.status = FINISHED
                        key_resp_T1.status = FINISHED
                if key_resp_T1.status == STARTED and not waitOnFlip:
                    theseKeys = key_resp_T1.getKeys(keyList=['f','j'], ignoreKeys=["escape"], waitRelease=False)
                    _key_resp_T1_allKeys.extend(theseKeys)
                    if len(_key_resp_T1_allKeys):
                        key_resp_T1.keys = _key_resp_T1_allKeys[-1].name  # just the last key pressed
                        key_resp_T1.rt = _key_resp_T1_allKeys[-1].rt
                        key_resp_T1.duration = _key_resp_T1_allKeys[-1].duration
                        # was this correct?
                        if (key_resp_T1.keys == str(responseT1)) or (key_resp_T1.keys == responseT1):
                            key_resp_T1.corr = 1
                        else:
                            key_resp_T1.corr = 0
                # Run 'Each Frame' code from code_text_T1ask
                if task == 'dual':
                    # send AskT1(?) trigger (50)
                    if text_T1ask.status == STARTED and not stimulus_pulse_started:
                        stimulus_pulse_started = True
                        stimulus_pulse_start_time = globalClock.getTime()
                        send_trigger(50)
                    #reset trigger
                    if stimulus_pulse_started and not stimulus_pulse_ended:
                        if globalClock.getTime() - stimulus_pulse_start_time >= 0.005:
                            send_trigger(0)
                            stimulus_pulse_ended = True
                
                    # send AnswerT1 trigger (51/52/53/54)
                    if key_resp_T1.keys and not pressed:
                        pressed = True
                        stimulus_pulse_start_time = globalClock.getTime()
                        if key_resp_T1.keys == responseT1: 
                            counter_T1 += 1
                            event_ans_T1 = 53 if responseT1 == 'j' else 51
                        else:
                            event_ans_T1 = 54 if responseT1 == 'j' else 52
                        send_trigger(event_ans_T1)
                    #reset trigger
                    if pressed:
                        if globalClock.getTime() - stimulus_pulse_start_time >= 0.005:
                            send_trigger(0)
                            pressed = False
                            continueRoutine = False
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    Ask_T1.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in Ask_T1.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "Ask_T1" ---
            for thisComponent in Ask_T1.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for Ask_T1
            Ask_T1.tStop = globalClock.getTime(format='float')
            Ask_T1.tStopRefresh = tThisFlipGlobal
            thisExp.addData('Ask_T1.stopped', Ask_T1.tStop)
            # check responses
            if key_resp_T1.keys in ['', [], None]:  # No response was made
                key_resp_T1.keys = None
                # was no response the correct answer?!
                if str(responseT1).lower() == 'none':
                   key_resp_T1.corr = 1;  # correct non-response
                else:
                   key_resp_T1.corr = 0;  # failed to respond (incorrectly)
            # store data for trial (TrialHandler)
            trial.addData('key_resp_T1.keys',key_resp_T1.keys)
            trial.addData('key_resp_T1.corr', key_resp_T1.corr)
            if key_resp_T1.keys != None:  # we had a response
                trial.addData('key_resp_T1.rt', key_resp_T1.rt)
                trial.addData('key_resp_T1.duration', key_resp_T1.duration)
            # Run 'End Routine' code from code_text_T1ask
            if task == 'dual':
                print(f'Corr T1: {key_resp_T1.corr}')
            
                if (trial.thisN+1)%10 == 0:
                    print(f'\n Accuracy T1: {counter_T1/10}\n')
                    counter_T1 = 0
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if Ask_T1.maxDurationReached:
                routineTimer.addTime(-Ask_T1.maxDuration)
            elif Ask_T1.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-2.000000)
            
            # --- Prepare to start Routine "blank_T1T2" ---
            # create an object to store info about Routine blank_T1T2
            blank_T1T2 = data.Routine(
                name='blank_T1T2',
                components=[text_blankT1T2],
            )
            blank_T1T2.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from code_trigger55
            trigger_sent = False
            trigger_reset = False
            # store start times for blank_T1T2
            blank_T1T2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            blank_T1T2.tStart = globalClock.getTime(format='float')
            blank_T1T2.status = STARTED
            thisExp.addData('blank_T1T2.started', blank_T1T2.tStart)
            blank_T1T2.maxDuration = None
            # skip Routine blank_T1T2 if its 'Skip if' condition is True
            blank_T1T2.skipped = continueRoutine and not (task == 'single')
            continueRoutine = blank_T1T2.skipped
            # keep track of which components have finished
            blank_T1T2Components = blank_T1T2.components
            for thisComponent in blank_T1T2.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "blank_T1T2" ---
            # if trial has changed, end Routine now
            if isinstance(trial, data.TrialHandler2) and thisTrial.thisN != trial.thisTrial.thisN:
                continueRoutine = False
            blank_T1T2.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 0.5:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *text_blankT1T2* updates
                
                # if text_blankT1T2 is starting this frame...
                if text_blankT1T2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text_blankT1T2.frameNStart = frameN  # exact frame index
                    text_blankT1T2.tStart = t  # local t and not account for scr refresh
                    text_blankT1T2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_blankT1T2, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_blankT1T2.started')
                    # update status
                    text_blankT1T2.status = STARTED
                    text_blankT1T2.setAutoDraw(True)
                
                # if text_blankT1T2 is active this frame...
                if text_blankT1T2.status == STARTED:
                    # update params
                    pass
                
                # if text_blankT1T2 is stopping this frame...
                if text_blankT1T2.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > text_blankT1T2.tStartRefresh + 0.5-frameTolerance:
                        # keep track of stop time/frame for later
                        text_blankT1T2.tStop = t  # not accounting for scr refresh
                        text_blankT1T2.tStopRefresh = tThisFlipGlobal  # on global time
                        text_blankT1T2.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'text_blankT1T2.stopped')
                        # update status
                        text_blankT1T2.status = FINISHED
                        text_blankT1T2.setAutoDraw(False)
                # Run 'Each Frame' code from code_trigger55
                #send trigger if missed answer (55)
                if not key_resp_T1.keys and not trigger_sent:
                    trigger_sent = True
                    stimulus_pulse_start_time = globalClock.getTime()
                    send_trigger(55)
                #reset trigger (0)
                if trigger_sent and not trigger_reset:
                    if globalClock.getTime() - stimulus_pulse_start_time >= 0.005:
                        send_trigger(0)
                        trigger_reset = True
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    blank_T1T2.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in blank_T1T2.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "blank_T1T2" ---
            for thisComponent in blank_T1T2.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for blank_T1T2
            blank_T1T2.tStop = globalClock.getTime(format='float')
            blank_T1T2.tStopRefresh = tThisFlipGlobal
            thisExp.addData('blank_T1T2.stopped', blank_T1T2.tStop)
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if blank_T1T2.maxDurationReached:
                routineTimer.addTime(-blank_T1T2.maxDuration)
            elif blank_T1T2.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-0.500000)
            
            # --- Prepare to start Routine "Ask_T2" ---
            # create an object to store info about Routine Ask_T2
            Ask_T2 = data.Routine(
                name='Ask_T2',
                components=[text_T2ask, key_resp_T2],
            )
            Ask_T2.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # create starting attributes for key_resp_T2
            key_resp_T2.keys = []
            key_resp_T2.rt = []
            _key_resp_T2_allKeys = []
            # Run 'Begin Routine' code from code_T2_ask
            pressed = False
            stimulus_pulse_started = False
            stimulus_pulse_ended = False
            # store start times for Ask_T2
            Ask_T2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            Ask_T2.tStart = globalClock.getTime(format='float')
            Ask_T2.status = STARTED
            thisExp.addData('Ask_T2.started', Ask_T2.tStart)
            Ask_T2.maxDuration = None
            # keep track of which components have finished
            Ask_T2Components = Ask_T2.components
            for thisComponent in Ask_T2.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "Ask_T2" ---
            # if trial has changed, end Routine now
            if isinstance(trial, data.TrialHandler2) and thisTrial.thisN != trial.thisTrial.thisN:
                continueRoutine = False
            Ask_T2.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 2.0:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *text_T2ask* updates
                
                # if text_T2ask is starting this frame...
                if text_T2ask.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                    # keep track of start time/frame for later
                    text_T2ask.frameNStart = frameN  # exact frame index
                    text_T2ask.tStart = t  # local t and not account for scr refresh
                    text_T2ask.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_T2ask, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_T2ask.started')
                    # update status
                    text_T2ask.status = STARTED
                    text_T2ask.setAutoDraw(True)
                
                # if text_T2ask is active this frame...
                if text_T2ask.status == STARTED:
                    # update params
                    pass
                
                # if text_T2ask is stopping this frame...
                if text_T2ask.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > text_T2ask.tStartRefresh + 2-frameTolerance:
                        # keep track of stop time/frame for later
                        text_T2ask.tStop = t  # not accounting for scr refresh
                        text_T2ask.tStopRefresh = tThisFlipGlobal  # on global time
                        text_T2ask.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'text_T2ask.stopped')
                        # update status
                        text_T2ask.status = FINISHED
                        text_T2ask.setAutoDraw(False)
                
                # *key_resp_T2* updates
                waitOnFlip = False
                
                # if key_resp_T2 is starting this frame...
                if key_resp_T2.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                    # keep track of start time/frame for later
                    key_resp_T2.frameNStart = frameN  # exact frame index
                    key_resp_T2.tStart = t  # local t and not account for scr refresh
                    key_resp_T2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(key_resp_T2, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'key_resp_T2.started')
                    # update status
                    key_resp_T2.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(key_resp_T2.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(key_resp_T2.clearEvents, eventType='keyboard')  # clear events on next screen flip
                
                # if key_resp_T2 is stopping this frame...
                if key_resp_T2.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > key_resp_T2.tStartRefresh + 2-frameTolerance:
                        # keep track of stop time/frame for later
                        key_resp_T2.tStop = t  # not accounting for scr refresh
                        key_resp_T2.tStopRefresh = tThisFlipGlobal  # on global time
                        key_resp_T2.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'key_resp_T2.stopped')
                        # update status
                        key_resp_T2.status = FINISHED
                        key_resp_T2.status = FINISHED
                if key_resp_T2.status == STARTED and not waitOnFlip:
                    theseKeys = key_resp_T2.getKeys(keyList=['f','j'], ignoreKeys=["escape"], waitRelease=False)
                    _key_resp_T2_allKeys.extend(theseKeys)
                    if len(_key_resp_T2_allKeys):
                        key_resp_T2.keys = _key_resp_T2_allKeys[-1].name  # just the last key pressed
                        key_resp_T2.rt = _key_resp_T2_allKeys[-1].rt
                        key_resp_T2.duration = _key_resp_T2_allKeys[-1].duration
                        # was this correct?
                        if (key_resp_T2.keys == str(responseT2)) or (key_resp_T2.keys == responseT2):
                            key_resp_T2.corr = 1
                        else:
                            key_resp_T2.corr = 0
                # Run 'Each Frame' code from code_T2_ask
                # send AskT2(?) trigger (60)
                if text_T2ask.status == STARTED and not stimulus_pulse_started:
                    stimulus_pulse_started = True
                    stimulus_pulse_start_time = globalClock.getTime()
                    send_trigger(60)
                #reset trigger
                if stimulus_pulse_started and not stimulus_pulse_ended:
                    if globalClock.getTime() - stimulus_pulse_start_time >= 0.005:
                        send_trigger(0)
                        stimulus_pulse_ended = True
                
                # send AnswerT1(61/62/63/64) trigger
                if key_resp_T2.keys and not pressed:
                    pressed = True
                    stimulus_pulse_start_time = globalClock.getTime()
                    if key_resp_T2.keys == responseT2: 
                        counter_T2 += 1
                        event_ans_T2 = 61 if related == 'related' else 63
                    else:
                        event_ans_T2 = 62 if related == 'related' else 64
                    send_trigger(event_ans_T2)
                #reset trigger
                if pressed:
                    if globalClock.getTime() - stimulus_pulse_start_time >= 0.005:
                        send_trigger(0)
                        pressed = False
                        continueRoutine = False
                
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    Ask_T2.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in Ask_T2.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "Ask_T2" ---
            for thisComponent in Ask_T2.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for Ask_T2
            Ask_T2.tStop = globalClock.getTime(format='float')
            Ask_T2.tStopRefresh = tThisFlipGlobal
            thisExp.addData('Ask_T2.stopped', Ask_T2.tStop)
            # check responses
            if key_resp_T2.keys in ['', [], None]:  # No response was made
                key_resp_T2.keys = None
                # was no response the correct answer?!
                if str(responseT2).lower() == 'none':
                   key_resp_T2.corr = 1;  # correct non-response
                else:
                   key_resp_T2.corr = 0;  # failed to respond (incorrectly)
            # store data for trial (TrialHandler)
            trial.addData('key_resp_T2.keys',key_resp_T2.keys)
            trial.addData('key_resp_T2.corr', key_resp_T2.corr)
            if key_resp_T2.keys != None:  # we had a response
                trial.addData('key_resp_T2.rt', key_resp_T2.rt)
                trial.addData('key_resp_T2.duration', key_resp_T2.duration)
            # Run 'End Routine' code from code_T2_ask
            print(f'Corr T2: {key_resp_T2.corr}')
            
            if (trial.thisN+1)%10 == 0:
                print(f'\n Accuracy T2: {counter_T2/10}')
                counter_T2 = 0
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if Ask_T2.maxDurationReached:
                routineTimer.addTime(-Ask_T2.maxDuration)
            elif Ask_T2.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-2.000000)
            
            # --- Prepare to start Routine "blank_2s" ---
            # create an object to store info about Routine blank_2s
            blank_2s = data.Routine(
                name='blank_2s',
                components=[text_blank2s],
            )
            blank_2s.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from code_trigger65
            trigger_sent = False
            trigger_reset = False
            # store start times for blank_2s
            blank_2s.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            blank_2s.tStart = globalClock.getTime(format='float')
            blank_2s.status = STARTED
            thisExp.addData('blank_2s.started', blank_2s.tStart)
            blank_2s.maxDuration = None
            # keep track of which components have finished
            blank_2sComponents = blank_2s.components
            for thisComponent in blank_2s.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "blank_2s" ---
            # if trial has changed, end Routine now
            if isinstance(trial, data.TrialHandler2) and thisTrial.thisN != trial.thisTrial.thisN:
                continueRoutine = False
            blank_2s.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 2.0:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *text_blank2s* updates
                
                # if text_blank2s is starting this frame...
                if text_blank2s.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text_blank2s.frameNStart = frameN  # exact frame index
                    text_blank2s.tStart = t  # local t and not account for scr refresh
                    text_blank2s.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_blank2s, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_blank2s.started')
                    # update status
                    text_blank2s.status = STARTED
                    text_blank2s.setAutoDraw(True)
                
                # if text_blank2s is active this frame...
                if text_blank2s.status == STARTED:
                    # update params
                    pass
                
                # if text_blank2s is stopping this frame...
                if text_blank2s.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > text_blank2s.tStartRefresh + 2-frameTolerance:
                        # keep track of stop time/frame for later
                        text_blank2s.tStop = t  # not accounting for scr refresh
                        text_blank2s.tStopRefresh = tThisFlipGlobal  # on global time
                        text_blank2s.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'text_blank2s.stopped')
                        # update status
                        text_blank2s.status = FINISHED
                        text_blank2s.setAutoDraw(False)
                # Run 'Each Frame' code from code_trigger65
                if not key_resp_T2.keys and not trigger_sent:
                    trigger_sent = True
                    stimulus_pulse_start_time = globalClock.getTime()
                    send_trigger(65)
                if trigger_sent and not trigger_reset:
                    if globalClock.getTime() - stimulus_pulse_start_time >= 0.005:
                        send_trigger(0)
                        trigger_reset = True
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    blank_2s.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in blank_2s.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "blank_2s" ---
            for thisComponent in blank_2s.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for blank_2s
            blank_2s.tStop = globalClock.getTime(format='float')
            blank_2s.tStopRefresh = tThisFlipGlobal
            thisExp.addData('blank_2s.stopped', blank_2s.tStop)
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if blank_2s.maxDurationReached:
                routineTimer.addTime(-blank_2s.maxDuration)
            elif blank_2s.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-2.000000)
            thisExp.nextEntry()
            
        # completed 1.0 repeats of 'trial'
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        # --- Prepare to start Routine "rest" ---
        # create an object to store info about Routine rest
        rest = data.Routine(
            name='rest',
            components=[text_rest, press_to_continue],
        )
        rest.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code_rest
        # skip rest after last block
        if block.thisN+1 == 12:
            continueRoutine = False
        else:
            stimulus_pulse_started = False
            stimulus_pulse_ended = False
        text_rest.setText(txt_rest)
        # create starting attributes for press_to_continue
        press_to_continue.keys = []
        press_to_continue.rt = []
        _press_to_continue_allKeys = []
        # store start times for rest
        rest.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        rest.tStart = globalClock.getTime(format='float')
        rest.status = STARTED
        thisExp.addData('rest.started', rest.tStart)
        rest.maxDuration = None
        # keep track of which components have finished
        restComponents = rest.components
        for thisComponent in rest.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "rest" ---
        # if trial has changed, end Routine now
        if isinstance(block, data.TrialHandler2) and thisBlock.thisN != block.thisTrial.thisN:
            continueRoutine = False
        rest.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # Run 'Each Frame' code from code_rest
            if block.thisN+1 != 12:
                #send rest trigger (100)
                if text_rest.status == STARTED and not stimulus_pulse_started:
                    stimulus_pulse_started = True
                    stimulus_pulse_start_time = globalClock.getTime()
                    send_trigger(100)
                #reset trigger (0)      
                if stimulus_pulse_started and not stimulus_pulse_ended:
                    if globalClock.getTime() - stimulus_pulse_start_time >= 0.005:
                        send_trigger(0)
                        stimulus_pulse_ended = True
            
            # *text_rest* updates
            
            # if text_rest is starting this frame...
            if text_rest.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_rest.frameNStart = frameN  # exact frame index
                text_rest.tStart = t  # local t and not account for scr refresh
                text_rest.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_rest, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_rest.started')
                # update status
                text_rest.status = STARTED
                text_rest.setAutoDraw(True)
            
            # if text_rest is active this frame...
            if text_rest.status == STARTED:
                # update params
                pass
            
            # *press_to_continue* updates
            waitOnFlip = False
            
            # if press_to_continue is starting this frame...
            if press_to_continue.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                press_to_continue.frameNStart = frameN  # exact frame index
                press_to_continue.tStart = t  # local t and not account for scr refresh
                press_to_continue.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(press_to_continue, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'press_to_continue.started')
                # update status
                press_to_continue.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(press_to_continue.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(press_to_continue.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if press_to_continue.status == STARTED and not waitOnFlip:
                theseKeys = press_to_continue.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _press_to_continue_allKeys.extend(theseKeys)
                if len(_press_to_continue_allKeys):
                    press_to_continue.keys = _press_to_continue_allKeys[-1].name  # just the last key pressed
                    press_to_continue.rt = _press_to_continue_allKeys[-1].rt
                    press_to_continue.duration = _press_to_continue_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                rest.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in rest.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "rest" ---
        for thisComponent in rest.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for rest
        rest.tStop = globalClock.getTime(format='float')
        rest.tStopRefresh = tThisFlipGlobal
        thisExp.addData('rest.stopped', rest.tStop)
        # check responses
        if press_to_continue.keys in ['', [], None]:  # No response was made
            press_to_continue.keys = None
        block.addData('press_to_continue.keys',press_to_continue.keys)
        if press_to_continue.keys != None:  # we had a response
            block.addData('press_to_continue.rt', press_to_continue.rt)
            block.addData('press_to_continue.duration', press_to_continue.duration)
        # the Routine "rest" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'block'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "end" ---
    # create an object to store info about Routine end
    end = data.Routine(
        name='end',
        components=[text_thanks],
    )
    end.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    text_thanks.setText(txt_thanks)
    # Run 'Begin Routine' code from code_end
    stimulus_pulse_started = False
    stimulus_pulse_ended = False
    # store start times for end
    end.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    end.tStart = globalClock.getTime(format='float')
    end.status = STARTED
    thisExp.addData('end.started', end.tStart)
    end.maxDuration = None
    # keep track of which components have finished
    endComponents = end.components
    for thisComponent in end.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "end" ---
    end.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 2.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_thanks* updates
        
        # if text_thanks is starting this frame...
        if text_thanks.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_thanks.frameNStart = frameN  # exact frame index
            text_thanks.tStart = t  # local t and not account for scr refresh
            text_thanks.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_thanks, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_thanks.started')
            # update status
            text_thanks.status = STARTED
            text_thanks.setAutoDraw(True)
        
        # if text_thanks is active this frame...
        if text_thanks.status == STARTED:
            # update params
            pass
        
        # if text_thanks is stopping this frame...
        if text_thanks.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_thanks.tStartRefresh + 2-frameTolerance:
                # keep track of stop time/frame for later
                text_thanks.tStop = t  # not accounting for scr refresh
                text_thanks.tStopRefresh = tThisFlipGlobal  # on global time
                text_thanks.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_thanks.stopped')
                # update status
                text_thanks.status = FINISHED
                text_thanks.setAutoDraw(False)
        # Run 'Each Frame' code from code_end
        #send finish trigger (200)
        if text_thanks.status == STARTED and not stimulus_pulse_started:
            stimulus_pulse_started = True
            stimulus_pulse_start_time = globalClock.getTime()
            send_trigger(200)
        #reset trigger (0)      
        if stimulus_pulse_started and not stimulus_pulse_ended:
            if globalClock.getTime() - stimulus_pulse_start_time >= 0.005:
                send_trigger(0)
                stimulus_pulse_ended = True
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            end.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in end.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "end" ---
    for thisComponent in end.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for end
    end.tStop = globalClock.getTime(format='float')
    end.tStopRefresh = tThisFlipGlobal
    thisExp.addData('end.stopped', end.tStop)
    # Run 'End Routine' code from code_end
    if port_type == 'serial':
        s_port.close()
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if end.maxDurationReached:
        routineTimer.addTime(-end.maxDuration)
    elif end.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-2.000000)
    thisExp.nextEntry()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
