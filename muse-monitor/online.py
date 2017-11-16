#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Management of data streams or other online processes.

Note that the module `threading` does not
"""

import utils

from serial.serialutil import SerialException
import threading

import pylsl as lsl


class LSLStreamer(threading.Thread):

    def __init__(self, inlet, data_buffer=None):
        threading.Thread.__init__(self)
        if data_buffer is None:
            data_buffer = LSLDataBuffer(inlet)
        self.data_buffer = data_buffer
        self.inlet = inlet

    def run(self):
        try:
            while True:
                samples, timestamps = self.inlet.pull_chunk(timeout=1.0,
                                                            max_samples=12)
                if timestamps:
                    timestamps = np.arange(len(timestamps), dtype=np.float64)
                    timestamps /= sfreq
                    timestamps += times[-1] + 1./sfreq
                    self.times = np.concatenate([self.times, timestamps])
                    self.times = self.times[-n_samples:]
                    self.data = np.vstack([self.data, samples])
                    self.data = self.data[-n_samples:]
                    with data_buffer.lock:
                        self.data_buffer.times = np.copy(self.times)
                        self.data_buffer.data = np.copy(self.data)
        except SerialException:
            print("EEG streaming interrupted. Device disconnected?")


class LSLDataBuffer(object):

    def __init__(self, inlet, window=5):
        self.inlet = inlet
        self.window = window
        sfreq = inlet.info.nominal_srate()
        n_chan = inlet.info.channel_count()
        n_samples = int(window * sfreq)
        self.data = np.zeros((n_samples, n_chan))
        self.times = np.arange(-window, 0, 1./sfreq)
        self.lock = threading.Lock()


def get_lsl_inlet(stream_type='EEG'):
    streams = lsl.resolve_stream('type', stream_type)
    try:
        inlet = lsl.StreamInlet(streams[0])
    except IndexError:
        raise IOError("No EEG stream resolved by LSL")
    return inlet
