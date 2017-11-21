#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Management of data streams or other online processes.
"""

from serial.serialutil import SerialException
import threading

import numpy as np
import pylsl as lsl


class LSLStreamer(threading.Thread):
    """

    """

    def __init__(self, inlet=None, dejitter=True, window=5, chunk_samples=12):
        threading.Thread.__init__(self)
        if inlet is None:
            inlet = self.get_lsl_inlet()
        self.inlet = inlet
        self.dejitter = dejitter
        self.window = window
        self.chunk_samples = chunk_samples

        # inlet parameters
        info = inlet.info()
        self.sfreq = info.nominal_srate()
        self.n_chan = info.channel_count()
        self.n_samples = int(window * self.sfreq)
        self.ch_names = LSLStreamer.get_ch_names(info)

        # thread control
        self.updated = threading.Event()
        self.lock = threading.Lock()

        # streaming data type and initialization
        self.dtype = np.dtype([('time', np.float64),
                               ('ch_values', np.float64, self.n_chan)])
        self.init_data()

    def run(self):
        try:
            while True:
                samples, timestamps = self.inlet.pull_chunk(timeout=1.0,
                                                            max_samples=self.chunk_samples)
                if timestamps:
                    if self.dejitter:
                        timestamps = self.dejitter_timestamps(timestamps)
                    new_data = np.array(list(zip(timestamps, samples)),
                                        dtype=self.dtype)
                    with self.lock:
                        self.new_data = new_data
                        self.__update_data()
                    self.updated.set()


        except SerialException:
            print("BGAPI streaming interrupted. Device disconnected?")

    def init_data(self):
        with self.lock:
            self.data = np.zeros((self.n_samples,), dtype=self.dtype)
            self.data['time'] = np.arange(-self.window, 0, 1./self.sfreq)

    def __update_data(self):
        """Append `new_data` to `data`and retain the `n_samples` newest samples."""
        self.data = np.concatenate([self.data, self.new_data], axis=0)
        self.data = self.data[-self.n_samples:]

    def dejitter_timestamps(self, timestamps):
        dejittered = np.arange(len(timestamps), dtype=np.float64)
        dejittered /= self.sfreq
        dejittered += self.data['time'][-1] + 1./self.sfreq
        return dejittered

    @staticmethod
    def get_lsl_inlet(stream_type='EEG'):
        streams = lsl.resolve_stream('type', stream_type)
        try:
            inlet = lsl.StreamInlet(streams[0])
        except IndexError:
            raise IOError("No stream resolved by LSL.")
        return inlet

    @staticmethod
    def get_ch_names(info):
        def next_ch_name():
            ch_xml = info.desc().child('channels').first_child()
            for ch in range(info.channel_count()):
                yield ch_xml.child_value('label')
                ch_xml = ch_xml.next_sibling()
        return list(next_ch_name())


class LSLDataStore(threading.Thread):

    def __init__(self, lsl_streamer):
        threading.Thread.__init__(self)
        self.streamer = lsl_streamer
        self.dtype = self.streamer.dtype
        self.init_data()

    def run(self):
        pass

    def store(self, time, relative_time=True):
        n_samples = int(time * self.streamer.sfreq)
        self.init_data()
        self.streamer.updated.clear()
        while len(self.data) < n_samples:
            self.streamer.updated.wait()
            with self.streamer.lock:
                self.__get_new_data()
                self.streamer.updated.clear()
        self.data = self.data[-n_samples:]
        if relative_time:
            self.data['time'] -= self.data['time'][0]

    def init_data(self):
        self.data = np.zeros(0, dtype=self.dtype)

    def __get_new_data(self):
        self.data = np.concatenate([self.data, self.streamer.new_data], axis=0)
