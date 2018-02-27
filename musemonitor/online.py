#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Management of data streams or other online processes.
"""

from musemonitor import utils

from serial.serialutil import SerialException
import threading

import numpy as np
import pylsl as lsl


class Data(threading.Thread):

    def __init__(self, metadata=None):
        self.metadata = metadata
        self.init()

        # thread control
        self.updated = threading.Event()
        self._lock = threading.Lock()

    def get(self):
        """Return copy of data window. Thread safe."""
        try:
            with self._lock:
                return np.copy(self._data)
        except AttributeError:
            raise NotImplementedError()

    def init(self):
        raise NotImplementedError()

    def update(self):
        raise NotImplementedError()


class TimeSeries(Data):
    """
    Attributes:
        window (float): Number of seconds of most recent data to store.
            Approximate, due to floor conversion to number of samples.
    """

    def __init__(self, ch_names, sfreq, window=10, record=True, metadata=None,
                 filename=None):
        Data.__init__(self, metadata)
        channels_dtype = np.dtype({'names': ch_names,
                                   'formats': ['f8'] * len(ch_names)})
        self._dtype = np.dtype([('time', 'f8'), ('channels', channels_dtype)])

        self.n_samples = int(window * self.sfreq)
        self._count = 0

        self.new = np.zeros(0, dtype=self._dtype)

        if filename is None:
            # TODO: mkdir if not exist
            filename = './data/data_{}.csv'.format(0) # TODO: increment

        self._file = file(filename, 'a')

    def init(self):
        """Initialize stored samples to zeros."""
        with self._lock:
            self._data = np.zeros((self.n_samples,), dtype=self._dtype)
            self._data['time'] = np.arange(-self.window, 0, 1./self.sfreq)

    def update(self, timestamps, samples):
        """Append most recent chunk to stored data and retain window size."""
        new = self._format_data(timestamps, samples)
        with self._lock:
            self._data = np.concatenate([self.data, new], axis=0)
            self._data = self.data[-self.n_samples:]  # TODO: remainder
            self.updated.set()
            self.count += len(new)
            if self.count > self.n_samples:
                self._write_to_file()
                self.count = 0  # TODO: remainder

    def _write_to_file(self):
        np.savetxt(self._file, self._data)

    def format_data_array(self, timestamps, samples):
        """Format data `numpy.ndarray` from timestamps and samples."""
        samples_tuples = [tuple(sample) for sample in samples]
        data_array = np.array(list(zip(timestamps, samples_tuples)),
                              dtype=self._dtype)
        return data_array

    @property
    def ch_names(self):
        """Names of channels."""
        return self._data.dtype['channels'].names

    @property
    def window(self):
        """Actual number of seconds stored.

        Not necessarily the same as the requested window size due to flooring
        to the nearest sample.
        """
        return self.n_samples / self.sfreq


class LSLStreamer(threading.Thread):
    """Stores most recent samples pulled from an LSL inlet.

    Attributes:
        inlet (pylsl.StreamInlet): The LSL inlet from which to stream data.
        data (numpy.ndarray): Most recent `n_samples` streamed from inlet.
            `'samples'` initialized to zeros, some of which will remain before
            the first `n_samples` have been streamed.
        new_data (numpy.ndarray): Data pulled in most recent chunk.
        dejitter (bool): Whether to regularize inter-sample intervals.
        sfreq (int): Sampling frequency of associated LSL inlet.
        n_chan (int): Number of channels in associated LSL inlet.
        n_samples (int): Number of most recent samples to store.

        updated (threading.Event): Flag when new data is pulled.
        lock (threading.Lock): Thread lock for safe access to streamed data.
        proceed (bool): Whether to keep streaming; set to False to end stream
            after current chunk.

    TODO:
        Exclude channels by name.
    """

    def __init__(self, inlet=None, data=None, dejitter=True,
                 chunk_samples=12, autostart=True):
        """Instantiate LSLStreamer given length of data store in seconds.

        Args:
            inlet (pylsl.StreamInlet): The LSL inlet from which to pull chunks.
                Defaults to call to `get_lsl_inlet`.
            dejitter (bool): Whether to regularize inter-sample intervals.
            chunk_samples (int): Maximum number of samples per chunk pulled.
            autostart (bool): Whether to start streaming on instantiation.
        """
        threading.Thread.__init__(self)
        if inlet is None:
            inlet = get_lsl_inlet()
        self.inlet = inlet
        self.dejitter = dejitter

        # inlet parameters
        info = inlet.info()
        self.sfreq = info.nominal_srate()
        self.n_chan = info.channel_count()
        self.ch_names = get_ch_names(info)

        # data class
        if data is None:
            self.data = TimeSeries(self.ch_names, self.sfreq, metadata=None)
        else:
            self.data = data

        # manual thread switch
        self.proceed = True

        # function aliases
        self._pull_chunk = lambda: inlet.pull_chunk(timeout=1.0,
                                                     max_samples=chunk_samples)

        if autostart:
            self.start()

    def run(self):
        """Streaming thread. Overrides `threading.Thread.run`."""
        try:
            while self.proceed:
                samples, timestamps = self._pull_chunk()
                if timestamps:
                    if self.dejitter:
                        timestamps = self._dejitter_timestamps(timestamps)
                    self.data.update(timestamps, samples)

        except SerialException:
            print("BGAPI streaming interrupted. Device disconnected?")


    def _dejitter_timestamps(self, timestamps):
        """Partial function for more concise call during loop."""
        dejittered = utils.dejitter_timestamps(timestamps, sfreq=self.sfreq,
                                               last_time=self.data['time'][-1])
        return dejittered


def get_lsl_inlet(stream_type='EEG'):
    """Resolve an LSL stream and return the corresponding inlet.

    Args:
        stream_type (str): Type of LSL stream to resolve.

    Returns:
        pylsl.StreamInlet: LSL inlet of resolved stream.
    """
    streams = lsl.resolve_stream('type', stream_type)
    try:
        inlet = lsl.StreamInlet(streams[0])
    except IndexError:
        raise IOError("No stream resolved by LSL.")
    return inlet


def get_ch_names(info):
    """Return the channel names associated with an LSL inlet.

    Args:
        info ():

    Returns:
        List[str]: Channel names.
    """
    def next_ch_name():
        ch_xml = info.desc().child('channels').first_child()
        for ch in range(info.channel_count()):
            yield ch_xml.child_value('label')
            ch_xml = ch_xml.next_sibling()
    return list(next_ch_name())
