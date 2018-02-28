#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Binary SVM classifier of concentrating vs. relaxating states from EEG data."""

from musemonitor import online
import gatt2lsl

if __name__ == '__main__':
    outlet_streamer = gatt2lsl.LSLOutletStreamer()

    streamer = online.LSLStreamer()
