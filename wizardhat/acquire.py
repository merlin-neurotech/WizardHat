#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Acquisition of data from streaming protocols, particularly LSL.

The `Lab Streaming Layer`_ (LSL) is a protocol for transmission of time series
measurements over local networks. A number of tools are available for handling
data streamed over LSL; please refer to its documentation for details.

Note:
    In the standard implementation of Python (CPython), thread execution is
    managed by the Global Interpreter Lock (GIL). Because of this, and in
    particular because of unavoidable background processes such as memory
    management/garbage collection, Python programs cannot guarantee real-time
    execution, but only approach it. Thus, at high process loads there may be
    momentary streaming lags.

.. _Lab Streaming Layer:
   https://github.com/sccn/labstreaminglayer
"""

from wizardhat import utils, data

from serial.serialutil import SerialException
import threading

import numpy as np
import pylsl as lsl


class LSLStreamer:
    """Passes data from an LSL stream to a `data.TimeSeries` object.

    Attributes:
        inlet (pylsl.StreamInlet): The LSL inlet from which to stream data.
        data (data.TimeSeries): Object in which the incoming data is stored,
            and which manages writing of that data to disk.
        sfreq (int): The nominal sampling frequency of the stream.
        n_chan (int): The number of channels in the stream.
        ch_names (List[str]): The names of the channels in the stream.

    """

    def __init__(self, inlet=None, data=None, dejitter=True,
                 chunk_samples=12, autostart=True):
        """Instantiate LSLStreamer given length of data store in seconds.

        Args:
            inlet (pylsl.StreamInlet): The LSL inlet from which to stream data.
                By default, this is created by resolving an available LSL
                stream through a call to `get_lsl_inlet`.
            data (data.TimeSeries): Object in which the incoming data is
                stored, and that manages writing of data to disk. By default,
                this is instantiated based on the channel names and nominal
                sampling frequency provided by the LSL inlet.
            dejitter (bool): Whether to regularize inter-sample intervals.
                If `True`, any timestamps returned by LSL are replaced by
                evenly-spaced timestamps based on the stream's nominal sampling
                frequency. Cannot be changed after instantiation due to the
                inconsistencies this would introduce in the resulting data.
            chunk_samples (int): Maximum number of samples per chunk pulled
                from the inlet.
            autostart (bool): Whether to start streaming on instantiation.

        """
        # resolve LSL stream if necessary
        if inlet is None:
            inlet = get_lsl_inlet()
        self.inlet = inlet

        # acquire inlet parameters
        info = inlet.info()
        self.sfreq = info.nominal_srate()
        self.n_chan = info.channel_count()
        self.ch_names = get_ch_names(info)

        # instantiate the `data.TimeSeries` instance if one is not provided
        if data is None:
            metadata = {"pipeline": [type(self).__name__]}
            self.data = data.TimeSeries(self.ch_names, self.sfreq,
                                        metadata=metadata)
        else:
            # user-defined instance
            self.data = data
            # TODO: do a test update to make sure it's a TimeSeries instance
            #try:
            #    test_samples = np.zeros(0, dtype=self.data.dtype)
            #    self.data.update([0], test_samples)
            #except

        # alias for `inlet.pull_chunk`
        self._pull_chunk = lambda: inlet.pull_chunk(timeout=1.0,
                                                    max_samples=chunk_samples)

        self._dejitter = dejitter
        self._new_thread()
        if autostart:
            self.start()

    def start(self):
        """Start data streaming.

        As the thread can only be started once and by default is started on
        instantiation of `TimeSeries`, this has no effect on streaming if
        called more than once without subsequent calls to `stop`.

        Samples between a call to `stop` and a subsequent call to `start` will
        be lost, leading to a discontinuity in the stored data.

        TODO:
            * TimeSeries effects (e.g. warn about discontinuity on restart)
        """
        try:
            self._thread.start()
        except RuntimeError:
            if self._thread.ident:
                # thread exists but has stopped; create and start a new thread
                self._new_thread()
                self._thread.start()
            else:
                print("Streaming has already started!")

    def stop(self):
        """Stop data streaming."""
        self._proceed = False

    def _stream(self):
        """Streaming thread."""
        try:
            while self._proceed:
                samples, timestamps = self._pull_chunk()
                if timestamps:
                    if self._dejitter:
                        timestamps = self._dejitter_timestamps(timestamps)
                    self.data.update(timestamps, samples)

        except SerialException:
            print("BGAPI streaming interrupted. Device disconnected?")

    def _new_thread(self):
        # break loop in `stream` to cause thread to return
        self._proceed = False
        # create new thread
        self._thread = threading.Thread(target=self._stream)
        self._proceed = True

    def _dejitter_timestamps(self, timestamps):
        """Partial function for more concise call during loop."""
        dejittered = dejitter_timestamps(timestamps, sfreq=self.sfreq,
                                         last_time=self.data.last['time'])
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


def dejitter_timestamps(timestamps, sfreq, last_time=None):
    """Convert timestamps to have regular sampling intervals.

       Args:
           timestamps (List[float]): A list of timestamps.
           sfreq (int): The sampling frequency.
           last_time (float): Time of the last sample preceding this set of
               timestamps. Defaults to `-1/sfreq`.
    """
    if last_time is None:
        last_time = -1/sfreq
    dejittered = np.arange(len(timestamps), dtype=np.float64)
    dejittered /= sfreq
    dejittered += last_time + 1/sfreq
    return dejittered
