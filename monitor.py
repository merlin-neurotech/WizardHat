#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Binary SVM classifier of concentrating vs. relaxating states from EEG data."""

from musemonitor import online
import gatt2lsl

if __name__ == '__main__':
    outlet_streamer = gatt2lsl.LSLOutletStreamer()
	#outlet_streamer = gatt2lsl.LSLOutletDummy() #Let's add default arg so people can run either dummy or real without commenting
	print('Connected')
    streamer = online.LSLStreamer()
