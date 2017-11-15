#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""

import utils

import threading
import pylsl as lsl

class EEGStreamer(threading.Thread):
    def __init__(self, window=5):
        threading.Thread.__init__(self)
        streams = resolve_stream('type', 'EEG')
        try:
            self.inlet = lsl.StreamInlet(streams[0])
        except IndexError:
            raise IOError("No EEG stream resolved by LSL")
        self.info = self.inlet.info()
        self.sfreq = self.info.nominal_srate()
        n_samples = int(window * sfreq)
        n_chan = self.info.channel_count()
        self.data = np.zeros((n_samples, n_chan))
        self.times = np.arange(-window, 0, 1./sfreq)
    def run(self):
        while True:
            samples, timestamps = self.inlet.pull_chunk(timeout=1.0,
                                                        max_samples=12)
            if timestamps:
                timestamps = np.arange(len(timestamps), dtype=np.float64)
                timestamps /= sfreq
                timestamps += times[-1] + 1/sfreq
                self.times = np.concatenate([self.times, timestamps])
                self.times = self.times[-n_samples:]
                self.data = np.vstack([self.data, samples])
                self.data = self.data[-n_samples:]
