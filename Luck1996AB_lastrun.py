﻿#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2024.1.5),
    on Fri Jul 18 16:56:20 2025
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
psychopyVersion = '2024.1.5'
expName = 'Luck1996AB'  # from the Builder filename that created this script
# information about this experiment
expInfo = {
    'participant': '000',
    'order [1: single-dual or 2: dual-single]': '1',
    'refresh rate (Hz)': '60',
    'port(noEEGtest/parallel/serial/cedrus/ghent)': 'noEEGtest',
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
_winSize = [1470, 956]
_loggingLevel = logging.getLevel('exp')
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']
    # override logging level
    _loggingLevel = logging.getLevel(
        prefs.piloting['pilotLoggingLevel']
    )

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
        originPath='/Users/alexandra/Luck1996_presentationCode/Luck1996AB_lastrun.py',
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
    # this outputs to the screen, not a file
    logging.console.setLevel(_loggingLevel)
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log', level=_loggingLevel)
    
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
            winType='pyglet', allowStencil=False,
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
            win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate
    win.mouseVisible = False
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
    
    ioSession = '1'
    if 'session' in expInfo:
        ioSession = str(expInfo['session'])
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
    
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # prevent components from auto-drawing
    win.stashAutoDraw()
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
        # flip the screen
        win.flip()
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # restore auto-drawn components
    win.retrieveAutoDraw()
    # reset any timers
    for timer in timers:
        timer.reset()


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
    # Run 'Begin Experiment' code from set_exp
    import random
    import pandas as pd
    
    #local path to language
    local_path = 'languages/'+expInfo['language']
    
    with open(f'{local_path}/rest.txt', 'r') as file:
        txt_rest = file.read()
    
    with open(f'{local_path}/thanks.txt', 'r') as file:
        txt_thanks = file.read()
    
    # duration in frames
    refreshRate = int(expInfo['refresh rate (Hz)'])
    dur_frames_stim = round(0.033 * refreshRate)
    dur_frames_blank = round(0.050 * refreshRate)
    
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
    # Run 'Begin Experiment' code from set_EEG
    # serial/parallel port address 
    port_address = expInfo['port address']
    port_type = expInfo['port(noEEGtest/parallel/serial/cedrus/ghent)']
    
    if port_type == 'noEEGtest': #for testing
        def send_trigger(triggerCode):
            print(triggerCode) 
        def send_trigger_now(triggerCode):
            print(triggerCode)
    elif port_type == 'parallel':
        from psychopy import parallel
        # create parallel port
        p_port = parallel.ParallelPort(address = port_address)
        #send pport trigger
        def send_trigger(triggerCode):
            win.callOnFlip(p_port.setData, triggerCode)
        def send_trigger_now(triggerCode):
            p_port.setData(triggerCode)
    elif port_type in ['serial','ghent']:
        import serial
        if port_type == 'ghent':
            # set up a serial port
            s_port = serial.Serial(port_address, baudrate = 115200)
            #send serial port trigger
            def send_trigger(triggerCode):
                win.callOnFlip(s_port.write, triggerCode.to_bytes(1, 'big'))
            def send_trigger_now(triggerCode):
                s_port.write(triggerCode.to_bytes(1, 'big'))
        elif port_type == 'serial':
            # set up a serial port
            s_port = serial.Serial(port = port_address)
            #send serial port trigger
            def send_trigger(triggerCode):
                win.callOnFlip(s_port.write, str.encode(str(triggerCode)))
            def send_trigger_now(triggerCode):
                s_port.write(str.encode(str(triggerCode)))
    elif port_type == 'cedrus':
        import pyxid2
        import time
        # get a list of all attached XID devices
        devices = pyxid2.get_xid_devices()
        dev = devices[0] # get the first device to use
        dev.set_pulse_duration(300)
        def send_trigger(triggerCode):
            win.callOnFlip(dev.activate_line, bitmask=triggerCode)
        def send_trigger_now(triggerCode):
            dev.activate_line(bitmask=triggerCode)
    
    # --- Initialize components for Routine "instructions" ---
    text_instructions = visual.TextStim(win=win, name='text_instructions',
        text=None,
        font='Open Sans',
        pos=(0, 0), height=0.02, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp = keyboard.Keyboard(deviceName='key_resp')
    
    # --- Initialize components for Routine "set_trial" ---
    
    # --- Initialize components for Routine "T0_context" ---
    context_word = visual.TextStim(win=win, name='context_word',
        text='',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, 1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    blankContext = visual.TextStim(win=win, name='blankContext',
        text=None,
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "distractors_T0" ---
    text_distr_T0 = visual.TextStim(win=win, name='text_distr_T0',
        text='',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='blue', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    text_blank_distractors_T0 = visual.TextStim(win=win, name='text_blank_distractors_T0',
        text=None,
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    
    # --- Initialize components for Routine "T1_number" ---
    text_T1 = visual.TextStim(win=win, name='text_T1',
        text='',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='blue', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    text_blank_number_T1 = visual.TextStim(win=win, name='text_blank_number_T1',
        text=None,
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "distractors_T1" ---
    text_distr_T1 = visual.TextStim(win=win, name='text_distr_T1',
        text='',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='blue', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    text_blank_distractors_T1 = visual.TextStim(win=win, name='text_blank_distractors_T1',
        text=None,
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "T2_probe" ---
    text_T2 = visual.TextStim(win=win, name='text_T2',
        text='',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='red', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    text_blank_T2_probe = visual.TextStim(win=win, name='text_blank_T2_probe',
        text=None,
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "distractors_T2" ---
    text_distr_T2 = visual.TextStim(win=win, name='text_distr_T2',
        text='',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='blue', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    text_blank_distractors_T2 = visual.TextStim(win=win, name='text_blank_distractors_T2',
        text=None,
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "blank_1s" ---
    text_blank1s = visual.TextStim(win=win, name='text_blank1s',
        text=None,
        font='Arial',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "Ask_T1" ---
    text_T1ask = visual.TextStim(win=win, name='text_T1ask',
        text='?',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, 1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    key_resp_T1 = keyboard.Keyboard(deviceName='key_resp_T1')
    
    # --- Initialize components for Routine "blank_T1T2" ---
    text_blankT1T2 = visual.TextStim(win=win, name='text_blankT1T2',
        text=None,
        font='Arial',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "Ask_T2" ---
    text_T2ask = visual.TextStim(win=win, name='text_T2ask',
        text='?',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, 1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    key_resp_T2 = keyboard.Keyboard(deviceName='key_resp_T2')
    
    # --- Initialize components for Routine "blank_2s" ---
    text_blank2s = visual.TextStim(win=win, name='text_blank2s',
        text=None,
        font='Arial',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "rest" ---
    text_rest = visual.TextStim(win=win, name='text_rest',
        text='',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    press_to_continue = keyboard.Keyboard(deviceName='press_to_continue')
    
    # --- Initialize components for Routine "end" ---
    text_thanks = visual.TextStim(win=win, name='text_thanks',
        text='',
        font='Arial',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
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
    block = data.TrialHandler(nReps=1.0, method='sequential', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions('task_type.xlsx'),
        seed=None, name='block')
    thisExp.addLoop(block)  # add the loop to the experiment
    thisBlock = block.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisBlock.rgb)
    if thisBlock != None:
        for paramName in thisBlock:
            globals()[paramName] = thisBlock[paramName]
    
    for thisBlock in block:
        currentLoop = block
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisBlock.rgb)
        if thisBlock != None:
            for paramName in thisBlock:
                globals()[paramName] = thisBlock[paramName]
        
        # --- Prepare to start Routine "set_block" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('set_block.started', globalClock.getTime(format='float'))
        # Run 'Begin Routine' code from set_exp
        # set blocks order
        task = task1 if expInfo['order [1: single-dual or 2: dual-single]']=='1' else task2
                
        # choose rows for this trial
        trials_in_block = related_idx[-n:] + unrelated_idx[-n:]
        shuffle(trials_in_block)
        related_idx = related_idx[:len(related_idx)-n]
        unrelated_idx = unrelated_idx[:len(unrelated_idx)-n]
        
        # log
        print(f'\nBlock #{block.thisN+1}\nTask: {task}\n')
        # keep track of which components have finished
        set_blockComponents = []
        for thisComponent in set_blockComponents:
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
        routineForceEnded = not continueRoutine
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
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in set_blockComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "set_block" ---
        for thisComponent in set_blockComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('set_block.stopped', globalClock.getTime(format='float'))
        # the Routine "set_block" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "instructions" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('instructions.started', globalClock.getTime(format='float'))
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
        # keep track of which components have finished
        instructionsComponents = [text_instructions, key_resp]
        for thisComponent in instructionsComponents:
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
        routineForceEnded = not continueRoutine
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
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in instructionsComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "instructions" ---
        for thisComponent in instructionsComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('instructions.stopped', globalClock.getTime(format='float'))
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
        trial = data.TrialHandler(nReps=1.0, method='sequential', 
            extraInfo=expInfo, originPath=-1,
            trialList=data.importConditions(f'{local_path}/wordPairs.xlsx', selection=trials_in_block),
            seed=None, name='trial')
        thisExp.addLoop(trial)  # add the loop to the experiment
        thisTrial = trial.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
        if thisTrial != None:
            for paramName in thisTrial:
                globals()[paramName] = thisTrial[paramName]
        
        for thisTrial in trial:
            currentLoop = trial
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
            )
            # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
            if thisTrial != None:
                for paramName in thisTrial:
                    globals()[paramName] = thisTrial[paramName]
            
            # --- Prepare to start Routine "set_trial" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('set_trial.started', globalClock.getTime(format='float'))
            # Run 'Begin Routine' code from code_trial
            distractors_trial = list(distractors_df.dist[dist_idx:dist_idx+18])
            i = 0 # distractor index
            print(f'\nTrial #{trial.thisN+1}')
            
            #triggers
            if task == 'single':
                trigger_T0 = 10
                trigger_T1 = 20 if responseT1 == 'j' else 21
                if related == 'unrelated':
                    trigger_T2 = 101 if lagT1 == 1 else (103 if lagT1 == 3 else 107)
                elif related == 'related':
                    trigger_T2 = 111 if lagT1 == 1 else (113 if lagT1 == 3 else 117)
            elif task == 'dual':
                trigger_T0 = 11
                trigger_T1 = 22 if responseT1 == 'j' else 23
                if related == 'unrelated':
                    trigger_T2 = 201 if lagT1 == 1 else (203 if lagT1 == 3 else 207)
                elif related == 'related':
                    trigger_T2 = 211 if lagT1 == 1 else (213 if lagT1 == 3 else 217)
            # keep track of which components have finished
            set_trialComponents = []
            for thisComponent in set_trialComponents:
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
            routineForceEnded = not continueRoutine
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
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in set_trialComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "set_trial" ---
            for thisComponent in set_trialComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('set_trial.stopped', globalClock.getTime(format='float'))
            # Run 'End Routine' code from code_trial
            dist_idx += 18
            # the Routine "set_trial" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "T0_context" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('T0_context.started', globalClock.getTime(format='float'))
            # Run 'Begin Routine' code from code_trigger_T0
            stimulus_pulse_started = False
            stimulus_pulse_ended = False
            context_word.setText(T0)
            # keep track of which components have finished
            T0_contextComponents = [context_word, blankContext]
            for thisComponent in T0_contextComponents:
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
            routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                # Run 'Each Frame' code from code_trigger_T0
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
                
                # *context_word* updates
                
                # if context_word is starting this frame...
                if context_word.status == NOT_STARTED and frameN >= 0.0:
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
                    if frameN >= (context_word.frameNStart + refreshRate):
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
                if blankContext.status == NOT_STARTED and frameN >= refreshRate:
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
                    if frameN >= (blankContext.frameNStart + refreshRate):
                        # keep track of stop time/frame for later
                        blankContext.tStop = t  # not accounting for scr refresh
                        blankContext.tStopRefresh = tThisFlipGlobal  # on global time
                        blankContext.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'blankContext.stopped')
                        # update status
                        blankContext.status = FINISHED
                        blankContext.setAutoDraw(False)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in T0_contextComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "T0_context" ---
            for thisComponent in T0_contextComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('T0_context.stopped', globalClock.getTime(format='float'))
            # the Routine "T0_context" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # set up handler to look after randomisation of conditions etc
            trials_T0_context = data.TrialHandler(nReps=lagT0-1, method='sequential', 
                extraInfo=expInfo, originPath=-1,
                trialList=[None],
                seed=None, name='trials_T0_context')
            thisExp.addLoop(trials_T0_context)  # add the loop to the experiment
            thisTrials_T0_context = trials_T0_context.trialList[0]  # so we can initialise stimuli with some values
            # abbreviate parameter names if possible (e.g. rgb = thisTrials_T0_context.rgb)
            if thisTrials_T0_context != None:
                for paramName in thisTrials_T0_context:
                    globals()[paramName] = thisTrials_T0_context[paramName]
            
            for thisTrials_T0_context in trials_T0_context:
                currentLoop = trials_T0_context
                thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                )
                # abbreviate parameter names if possible (e.g. rgb = thisTrials_T0_context.rgb)
                if thisTrials_T0_context != None:
                    for paramName in thisTrials_T0_context:
                        globals()[paramName] = thisTrials_T0_context[paramName]
                
                # --- Prepare to start Routine "distractors_T0" ---
                continueRoutine = True
                # update component parameters for each repeat
                thisExp.addData('distractors_T0.started', globalClock.getTime(format='float'))
                # Run 'Begin Routine' code from code_trigger_distrT0
                stimulus_pulse_started = False
                stimulus_pulse_ended = False
                # Run 'Begin Routine' code from code_save_distrT0
                thisExp.addData('distractor_T0',distractors_trial[i])
                text_distr_T0.setText(distractors_trial[i])
                # keep track of which components have finished
                distractors_T0Components = [text_distr_T0, text_blank_distractors_T0]
                for thisComponent in distractors_T0Components:
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
                routineForceEnded = not continueRoutine
                while continueRoutine:
                    # get current time
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                    # update/draw components on each frame
                    # Run 'Each Frame' code from code_trigger_distrT0
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
                    
                    # check if all components have finished
                    if not continueRoutine:  # a component has requested a forced-end of Routine
                        routineForceEnded = True
                        break
                    continueRoutine = False  # will revert to True if at least one component still running
                    for thisComponent in distractors_T0Components:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished
                    
                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()
                
                # --- Ending Routine "distractors_T0" ---
                for thisComponent in distractors_T0Components:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                thisExp.addData('distractors_T0.stopped', globalClock.getTime(format='float'))
                # Run 'End Routine' code from code_save_distrT0
                i += 1
                # the Routine "distractors_T0" was not non-slip safe, so reset the non-slip timer
                routineTimer.reset()
                thisExp.nextEntry()
                
                if thisSession is not None:
                    # if running in a Session with a Liaison client, send data up to now
                    thisSession.sendExperimentData()
            # completed lagT0-1 repeats of 'trials_T0_context'
            
            
            # --- Prepare to start Routine "T1_number" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('T1_number.started', globalClock.getTime(format='float'))
            # Run 'Begin Routine' code from code_trigger_T1
            stimulus_pulse_started = False
            stimulus_pulse_ended = False
            text_T1.setText(str(int(T1)))
            # keep track of which components have finished
            T1_numberComponents = [text_T1, text_blank_number_T1]
            for thisComponent in T1_numberComponents:
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
            routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                # Run 'Each Frame' code from code_trigger_T1
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
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in T1_numberComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "T1_number" ---
            for thisComponent in T1_numberComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('T1_number.stopped', globalClock.getTime(format='float'))
            # the Routine "T1_number" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # set up handler to look after randomisation of conditions etc
            trials_T1 = data.TrialHandler(nReps=lagT1-1, method='sequential', 
                extraInfo=expInfo, originPath=-1,
                trialList=[None],
                seed=None, name='trials_T1')
            thisExp.addLoop(trials_T1)  # add the loop to the experiment
            thisTrials_T1 = trials_T1.trialList[0]  # so we can initialise stimuli with some values
            # abbreviate parameter names if possible (e.g. rgb = thisTrials_T1.rgb)
            if thisTrials_T1 != None:
                for paramName in thisTrials_T1:
                    globals()[paramName] = thisTrials_T1[paramName]
            
            for thisTrials_T1 in trials_T1:
                currentLoop = trials_T1
                thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                )
                # abbreviate parameter names if possible (e.g. rgb = thisTrials_T1.rgb)
                if thisTrials_T1 != None:
                    for paramName in thisTrials_T1:
                        globals()[paramName] = thisTrials_T1[paramName]
                
                # --- Prepare to start Routine "distractors_T1" ---
                continueRoutine = True
                # update component parameters for each repeat
                thisExp.addData('distractors_T1.started', globalClock.getTime(format='float'))
                # Run 'Begin Routine' code from code_trigger_distrT1
                thisExp.addData('distractor_T1',distractors_trial[i])
                stimulus_pulse_started = False
                stimulus_pulse_ended = False
                text_distr_T1.setText(distractors_trial[i])
                # keep track of which components have finished
                distractors_T1Components = [text_distr_T1, text_blank_distractors_T1]
                for thisComponent in distractors_T1Components:
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
                routineForceEnded = not continueRoutine
                while continueRoutine:
                    # get current time
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                    # update/draw components on each frame
                    # Run 'Each Frame' code from code_trigger_distrT1
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
                    
                    # check if all components have finished
                    if not continueRoutine:  # a component has requested a forced-end of Routine
                        routineForceEnded = True
                        break
                    continueRoutine = False  # will revert to True if at least one component still running
                    for thisComponent in distractors_T1Components:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished
                    
                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()
                
                # --- Ending Routine "distractors_T1" ---
                for thisComponent in distractors_T1Components:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                thisExp.addData('distractors_T1.stopped', globalClock.getTime(format='float'))
                # Run 'End Routine' code from code_trigger_distrT1
                i += 1
                # the Routine "distractors_T1" was not non-slip safe, so reset the non-slip timer
                routineTimer.reset()
                thisExp.nextEntry()
                
                if thisSession is not None:
                    # if running in a Session with a Liaison client, send data up to now
                    thisSession.sendExperimentData()
            # completed lagT1-1 repeats of 'trials_T1'
            
            
            # --- Prepare to start Routine "T2_probe" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('T2_probe.started', globalClock.getTime(format='float'))
            # Run 'Begin Routine' code from code_trigger_T2
            stimulus_pulse_started = False
            stimulus_pulse_ended = False
            text_T2.setText(T2)
            # keep track of which components have finished
            T2_probeComponents = [text_T2, text_blank_T2_probe]
            for thisComponent in T2_probeComponents:
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
            routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                # Run 'Each Frame' code from code_trigger_T2
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
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in T2_probeComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "T2_probe" ---
            for thisComponent in T2_probeComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('T2_probe.stopped', globalClock.getTime(format='float'))
            # the Routine "T2_probe" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # set up handler to look after randomisation of conditions etc
            trials_T2 = data.TrialHandler(nReps=lagT2, method='sequential', 
                extraInfo=expInfo, originPath=-1,
                trialList=[None],
                seed=None, name='trials_T2')
            thisExp.addLoop(trials_T2)  # add the loop to the experiment
            thisTrials_T2 = trials_T2.trialList[0]  # so we can initialise stimuli with some values
            # abbreviate parameter names if possible (e.g. rgb = thisTrials_T2.rgb)
            if thisTrials_T2 != None:
                for paramName in thisTrials_T2:
                    globals()[paramName] = thisTrials_T2[paramName]
            
            for thisTrials_T2 in trials_T2:
                currentLoop = trials_T2
                thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                )
                # abbreviate parameter names if possible (e.g. rgb = thisTrials_T2.rgb)
                if thisTrials_T2 != None:
                    for paramName in thisTrials_T2:
                        globals()[paramName] = thisTrials_T2[paramName]
                
                # --- Prepare to start Routine "distractors_T2" ---
                continueRoutine = True
                # update component parameters for each repeat
                thisExp.addData('distractors_T2.started', globalClock.getTime(format='float'))
                # Run 'Begin Routine' code from code_trigger_distrT2
                thisExp.addData('distractor_T2',distractors_trial[i])
                stimulus_pulse_started = False
                stimulus_pulse_ended = False
                text_distr_T2.setText(distractors_trial[i])
                # keep track of which components have finished
                distractors_T2Components = [text_distr_T2, text_blank_distractors_T2]
                for thisComponent in distractors_T2Components:
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
                routineForceEnded = not continueRoutine
                while continueRoutine:
                    # get current time
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                    # update/draw components on each frame
                    # Run 'Each Frame' code from code_trigger_distrT2
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
                    
                    # check if all components have finished
                    if not continueRoutine:  # a component has requested a forced-end of Routine
                        routineForceEnded = True
                        break
                    continueRoutine = False  # will revert to True if at least one component still running
                    for thisComponent in distractors_T2Components:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished
                    
                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()
                
                # --- Ending Routine "distractors_T2" ---
                for thisComponent in distractors_T2Components:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                thisExp.addData('distractors_T2.stopped', globalClock.getTime(format='float'))
                # Run 'End Routine' code from code_trigger_distrT2
                i += 1
                # the Routine "distractors_T2" was not non-slip safe, so reset the non-slip timer
                routineTimer.reset()
                thisExp.nextEntry()
                
                if thisSession is not None:
                    # if running in a Session with a Liaison client, send data up to now
                    thisSession.sendExperimentData()
            # completed lagT2 repeats of 'trials_T2'
            
            
            # --- Prepare to start Routine "blank_1s" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('blank_1s.started', globalClock.getTime(format='float'))
            # keep track of which components have finished
            blank_1sComponents = [text_blank1s]
            for thisComponent in blank_1sComponents:
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
            routineForceEnded = not continueRoutine
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
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in blank_1sComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "blank_1s" ---
            for thisComponent in blank_1sComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('blank_1s.stopped', globalClock.getTime(format='float'))
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if routineForceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-1.000000)
            
            # --- Prepare to start Routine "Ask_T1" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('Ask_T1.started', globalClock.getTime(format='float'))
            # skip this Routine if its 'Skip if' condition is True
            continueRoutine = continueRoutine and not (task == 'single')
            # Run 'Begin Routine' code from code_text_T1ask
            if task == 'dual':
                pressed = False
                stimulus_pulse_started = False
                stimulus_pulse_ended = False
            # create starting attributes for key_resp_T1
            key_resp_T1.keys = []
            key_resp_T1.rt = []
            _key_resp_T1_allKeys = []
            # keep track of which components have finished
            Ask_T1Components = [text_T1ask, key_resp_T1]
            for thisComponent in Ask_T1Components:
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
            routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 2.0:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
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
                        send_trigger_now(event_ans_T1)
                    #reset trigger
                    if pressed:
                        if globalClock.getTime() - stimulus_pulse_start_time >= 0.005:
                            send_trigger_now(0)
                            pressed = False
                            continueRoutine = False
                
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
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in Ask_T1Components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "Ask_T1" ---
            for thisComponent in Ask_T1Components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('Ask_T1.stopped', globalClock.getTime(format='float'))
            # Run 'End Routine' code from code_text_T1ask
            if task == 'dual':
                print(f'Corr T1: {key_resp_T1.corr}')
            
                if (trial.thisN+1)%10 == 0:
                    print(f'\n Accuracy T1: {counter_T1/10}\n')
                    counter_T1 = 0
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
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if routineForceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-2.000000)
            
            # --- Prepare to start Routine "blank_T1T2" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('blank_T1T2.started', globalClock.getTime(format='float'))
            # skip this Routine if its 'Skip if' condition is True
            continueRoutine = continueRoutine and not (task == 'single')
            # Run 'Begin Routine' code from code_trigger55
            trigger_sent = False
            trigger_reset = False
            # keep track of which components have finished
            blank_T1T2Components = [text_blankT1T2]
            for thisComponent in blank_T1T2Components:
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
            routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 0.5:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
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
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in blank_T1T2Components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "blank_T1T2" ---
            for thisComponent in blank_T1T2Components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('blank_T1T2.stopped', globalClock.getTime(format='float'))
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if routineForceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-0.500000)
            
            # --- Prepare to start Routine "Ask_T2" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('Ask_T2.started', globalClock.getTime(format='float'))
            # Run 'Begin Routine' code from code_T2_ask
            pressed = False
            stimulus_pulse_started = False
            stimulus_pulse_ended = False
            # create starting attributes for key_resp_T2
            key_resp_T2.keys = []
            key_resp_T2.rt = []
            _key_resp_T2_allKeys = []
            # keep track of which components have finished
            Ask_T2Components = [text_T2ask, key_resp_T2]
            for thisComponent in Ask_T2Components:
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
            routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 2.0:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
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
                    send_trigger_now(event_ans_T2)
                #reset trigger
                if pressed:
                    if globalClock.getTime() - stimulus_pulse_start_time >= 0.005:
                        send_trigger_now(0)
                        pressed = False
                        continueRoutine = False
                
                
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
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in Ask_T2Components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "Ask_T2" ---
            for thisComponent in Ask_T2Components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('Ask_T2.stopped', globalClock.getTime(format='float'))
            # Run 'End Routine' code from code_T2_ask
            print(f'Corr T2: {key_resp_T2.corr}')
            
            if (trial.thisN+1)%10 == 0:
                print(f'\n Accuracy T2: {counter_T2/10}')
                counter_T2 = 0
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
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if routineForceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-2.000000)
            
            # --- Prepare to start Routine "blank_2s" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('blank_2s.started', globalClock.getTime(format='float'))
            # Run 'Begin Routine' code from code_trigger65
            trigger_sent = False
            trigger_reset = False
            # keep track of which components have finished
            blank_2sComponents = [text_blank2s]
            for thisComponent in blank_2sComponents:
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
            routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 2.0:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                # Run 'Each Frame' code from code_trigger65
                if not key_resp_T2.keys and not trigger_sent:
                    trigger_sent = True
                    stimulus_pulse_start_time = globalClock.getTime()
                    send_trigger(65)
                if trigger_sent and not trigger_reset:
                    if globalClock.getTime() - stimulus_pulse_start_time >= 0.005:
                        send_trigger(0)
                        trigger_reset = True
                
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
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in blank_2sComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "blank_2s" ---
            for thisComponent in blank_2sComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('blank_2s.stopped', globalClock.getTime(format='float'))
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if routineForceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-2.000000)
            thisExp.nextEntry()
            
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
        # completed 1.0 repeats of 'trial'
        
        
        # --- Prepare to start Routine "rest" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('rest.started', globalClock.getTime(format='float'))
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
        # keep track of which components have finished
        restComponents = [text_rest, press_to_continue]
        for thisComponent in restComponents:
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
        routineForceEnded = not continueRoutine
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
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in restComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "rest" ---
        for thisComponent in restComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('rest.stopped', globalClock.getTime(format='float'))
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
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed 1.0 repeats of 'block'
    
    
    # --- Prepare to start Routine "end" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('end.started', globalClock.getTime(format='float'))
    # Run 'Begin Routine' code from code_end
    stimulus_pulse_started = False
    stimulus_pulse_ended = False
    text_thanks.setText(txt_thanks)
    # keep track of which components have finished
    endComponents = [text_thanks]
    for thisComponent in endComponents:
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
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 2.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
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
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in endComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "end" ---
    for thisComponent in endComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('end.stopped', globalClock.getTime(format='float'))
    # Run 'End Routine' code from code_end
    if port_type == 'serial':
        s_port.close()
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
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
    # mark experiment handler as finished
    thisExp.status = FINISHED
    # shut down eyetracker, if there is one
    if deviceManager.getDevice('eyetracker') is not None:
        deviceManager.removeDevice('eyetracker')
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
    # shut down eyetracker, if there is one
    if deviceManager.getDevice('eyetracker') is not None:
        deviceManager.removeDevice('eyetracker')
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
