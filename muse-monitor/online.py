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

    def __init__(self, inlet=None, window=5, dejitter=True, chunk_samples=12,
                 autostart=True):
        """Instantiate LSLStreamer given length of data store in seconds.

        Args:
            inlet (pylsl.StreamInlet): The LSL inlet from which to pull chunks.
                Defaults to call to `get_lsl_inlet`.
            window (float): Number of seconds of most recent data to store.
                Approximate, due to floor conversion to number of samples.
            dejitter (bool): Whether to regularize inter-sample intervals.
            chunk_samples (int): Maximum number of samples per chunk pulled.
            autostart (bool): Whether to start streaming on instantiation.
        """
        threading.Thread.__init__(self)
        if inlet is None:
            inlet = self.get_lsl_inlet()
        self.inlet = inlet
        self.dejitter = dejitter

        # inlet parameters
        info = inlet.info()
        self.sfreq = info.nominal_srate()
        self.n_samples = int(window * self.sfreq)
        self.n_chan = info.channel_count()

        # thread control
        self.updated = threading.Event()
        self.lock = threading.Lock()
        self.proceed = True

        # data type and initialization
        channels_dtype = np.dtype({'names': get_ch_names(info),
                                  'formats': ['f8'] * self.n_chan})
        self.__dtype = np.dtype([('time', 'f8'), ('channels', channels_dtype)])
        self.init_data()
        self.new_data = np.zeros(0, dtype=self.__dtype)

        # function aliases
        self.__pull_chunk = lambda: inlet.pull_chunk(timeout=1.0,
                                                     max_samples=chunk_samples)

        if autostart:
            self.start()

    def run(self):
        """Streaming thread. Overrides `threading.Thread.run`."""
        try:
            while self.proceed:
                samples, timestamps = self.__pull_chunk()
                if timestamps:
                    if self.dejitter:
                        timestamps = self.__dejitter_timestamps(timestamps)
                    new_data = self.__format_data_array(timestamps, samples)
                    with self.lock:
                        self.new_data = new_data
                        self.__update_data()
                    self.updated.set()

        except SerialException:
            print("BGAPI streaming interrupted. Device disconnected?")

    def init_data(self):
        """Initialize stored samples to zeros."""
        with self.lock:
            self.data = np.zeros((self.n_samples,), dtype=self.__dtype)
            self.data['time'] = np.arange(-self.window, 0, 1./self.sfreq)

    def __dejitter_timestamps(self, timestamps):
        """Partial function for more concise call during loop."""
        dejittered = utils.dejitter_timestamps(timestamps, sfreq=self.sfreq,
                                               last_time=self.data['time'][-1])
        return dejittered

    def __format_data_array(self, timestamps, samples):
        """Format data `numpy.ndarray` from timestamps and samples."""
        samples_tuples = [tuple(sample) for sample in samples]
        data_array = np.array(list(zip(timestamps, samples_tuples)),
                            dtype=self.__dtype)
        return data_array

    def __update_data(self):
        """Append most recent chunk to stored data and retain window size."""
        self.data = np.concatenate([self.data, self.new_data], axis=0)
        self.data = self.data[-self.n_samples:]

    @property
    def ch_names(self):
        """Names of channels from associated LSL inlet."""
        return self.data.dtype['channels'].names

    @property
    def window(self):
        """Actual number of seconds stored.

        Not necessarily the same as the requested window size due to flooring
        to the nearest sample.
        """
        return self.n_samples / self.sfreq


class LSLRecorder(threading.Thread):
    """Make recordings of arbitrary length from a given `LSLStreamer`.

    Attributes:
        streamer (LSLStreamer): Associated LSL data streamer.
    """

    def __init__(self, lsl_streamer):
        """Initialize given an `LSLStreamer` instance.

        Args:
            lsl_streamer (LSLStreamer): LSL data streamer from which to record.
        """
        threading.Thread.__init__(self)
        self.streamer = lsl_streamer
        self.__dtype = self.streamer.data.dtype  # lock?

        self.init_data()

    def run(self):
        pass

    def record(self, length, relative_time=True):
        """Store the next `length` seconds of streamed data.

        Args:
            length (float): Number of seconds of data to store.
            relative_time (bool): Whether to set sample times relative to
                start of recording.
        """
        n_samples = int(length * self.streamer.sfreq)
        self.init_data()
        self.streamer.updated.clear()  # skip chunk pulled before
        while self.data.shape[0] < n_samples:
            self.streamer.updated.wait()
            with self.streamer.lock:
                self.__get_new_data()
                self.streamer.updated.clear()
        self.data = self.data[-n_samples:]
        if relative_time:
            self.data['time'] -= self.data['time'][0]

    def record_trial(self, spec, **kwargs):
        """Call `record` preceded by a message and a prompt.

        Args:
            spec (dict): Specification for trial.
               Contains the message and recording length.
            **kwargs: Keyword arguments to `record`.

        Returns:
            Instance's stored data upon completion of recording.
        """
        if 'msg' in spec:
            print(spec['msg'])
        print('Press Enter when ready to start recording.')
        input()
        self.record(spec['length'], **kwargs)
        return self.data

    def record_trials(self, specs, **kwargs):
        """Record multiple trials with `record_trial`.

        Args:
            specs (List[dict]): List of trial specifications.
            **kwargs: Keyword arguments to `record_trial`.

        Returns:
            A `dict` with keys as `'label'` values from individual trial
            specification dictionaries, and values as `numpy.ndarrays`
            for the corresponding recordings.
        """
        trials = {spec['label']: self.record_trial(spec, **kwargs)
                  for spec in specs}
        return trials

    def init_data(self):
        """Initialize data store as empty."""
        self.data = np.zeros(0, dtype=self.__dtype)

    def __get_new_data(self):
        self.data = np.concatenate([self.data, self.streamer.new_data], axis=0)


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
