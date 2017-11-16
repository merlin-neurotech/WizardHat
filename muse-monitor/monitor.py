#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""
import numpy as np

import online
import gui


if __name__ == '__main__':
    streamer = online.LSLStreamer(online.get_lsl_inlet())
    #gui.init_eegplot()

    #streamer.start() # start EEG streamer thread
    #streamer.join() # wait for EEG streamer to stop streaming
