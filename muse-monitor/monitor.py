#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""
import numpy as np

import online
import gui


if __name__ == '__main__':
    streamer = online.LSLStreamer()
    recorder = online.LSLRecorder(streamer)

    # initial messages
    welcome = """\nEEG data will be recorded. During each trial, please be as
    still as possible while following each set of instructions.\n"""
    print(welcome)

    # record trials
    specs = [{'length': 60,
              'msg': "Please focus visually and concentrate."},
             {'length': 60,
              'msg': "Please close your eyes and relax."}]
    trials = recorder.record_trials(specs)

    #gui.init_eegplot()

    #streamer.start() # start EEG streamer thread
    #streamer.join() # wait for EEG streamer to stop streaming
