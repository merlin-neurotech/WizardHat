#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Management of data streams or other online processes.
"""

import utils

from serial.serialutil import SerialException
import threading

import numpy as np
import pylsl as lsl


class LSLStreamer(threading.Thread):

    def __init__(self, inlet, data_buffer=None, dejitter=True):
        threading.Thread.__init__(self)
        if data_buffer is None:
            data_buffer = LSLDataBuffer(inlet)
        self.data_buffer = data_buffer
        self.inlet = inlet
        self.dejitter = dejitter
        self.sfreq = self.data_buffer.sfreq
        self.n_samples = self.data_buffer.n_samples

    def run(self):
        try:
            while True:
                samples, timestamps = self.inlet.pull_chunk(timeout=1.0,
                                                            max_samples=12)
                if timestamps:
                    if self.dejitter:
                        timestamps = self.dejitter(timestamps, self.last_time)
                    self.update_data(timestamps, samples)
                    with self.data_buffer.lock:
                        self.data_buffer.times = np.copy(self.times)
                        self.data_buffer.data = np.copy(self.samples)

        except SerialException:
            print("BGAPI streaming interrupted. Device disconnected?")

    def dejitter_timestamps(self, timestamps, last_time):
        dejittered = np.arange(len(timestamps), dtype=np.float64)
        dejittered /= self.sfreq
        dejittered += last_time + 1./self.sfreq
        return dejittered

    def updated_data(self, timestamps, samples):
        times = np.concatenate([self.times, timestamps])
        times = self.times[-self.n_samples:]
        samples = np.vstack([self.samples, samples])
        samples = self.samples[-self.n_samples:]
        return times, samples

    @property
    def last_time(self):
        return self.data_buffer.times[-1]

class LSLDataBuffer(object):

    def __init__(self, inlet, window=5):
        self.inlet = inlet
        self.window = window

        info = inlet.info()
        self.sfreq = info.nominal_srate()
        self.n_chan = info.channel_count()
        self.n_samples = int(window * self.sfreq)
        self.ch_names = list(self.__ch_names(info))

        self.zero_data()

        self.lock = threading.Lock()

    def zero_data(self):
        self.samples = np.zeros((self.n_samples, self.n_chan))
        self.times = np.arange(-self.window, 0, 1./self.sfreq)

    def __ch_names(self, info):
        ch_xml = info.desc().child('channels').first_child()
        for ch in range(self.n_chan):
            yield ch_xml.child_value('label')
            ch_xml = ch_xml.next_sibling()


def get_lsl_inlet(stream_type='EEG'):
    streams = lsl.resolve_stream('type', stream_type)
    try:
        inlet = lsl.StreamInlet(streams[0])
    except IndexError:
        raise IOError("No EEG stream resolved by LSL")
    return inlet
