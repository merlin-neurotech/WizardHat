""" """

import gatt2lsl
import musemonitor.online as online
import musemonitor.utils as utils

import time

import numpy as np

threshold = 40

outlet_streamer = gatt2lsl.LSLOutletStreamer()

streamer = online.LSLStreamer(window=1)
recorder = online.LSLRecorder(streamer)

ica = utils.ICACleanup(streamer.sfreq, streamer.ch_names)

while True:
    time.sleep(1)
    data = streamer.get_data()
    data = data["channels"].view((data.dtype["channels"][0], len(data.dtype["channels"].names)))
    m = np.mean(np.abs(data))
    msg = 1 if m > threshold else 0
    print(msg)
