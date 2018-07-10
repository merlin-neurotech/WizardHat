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

TODO:
    * dejitter_timestamps may be irrelevant depending on ble2lsl operation

.. _Lab Streaming Layer:
   https://github.com/sccn/labstreaminglayer
"""

from wizardhat import data

from serial.serialutil import SerialException
import threading

import numpy as np
import pylsl as lsl


class Acquirer:
    """Passes data from an LSL stream to a `data.TimeSeries` object.

    Attributes:
        inlets (dict[pylsl.StreamInlet]): The LSL inlet(s) to acquire.
        data (data.TimeSeries): Object in which the incoming data is stored,
            and which manages writing of that data to disk.
        sfreq (int): The nominal sampling frequency of the stream.
        n_chan (int): The number of channels in the stream.
        ch_names (List[str]): The names of the channels in the stream.

    TODO:
        * custom Data classes on instantiation? or pass config (e.g. windows)
        * partial acquisition if some streams don't conform (ValueErrors in
          init)
        * allow inlets/data to be referenced by aliases (i.e STREAMS from
          ble2lsl.devices)
        * change chunk samples to reflect variation?
    """

    def __init__(self, inlets=None, with_name='', dejitter=True,
                 chunk_samples=12, autostart=True, **kwargs):
        """Instantiate LSLStreamer given length of data store in seconds.

        Args:
            inlets (Iterable[pylsl.StreamInlet]): The LSL inlet(s) to acquire.
                By default, all available inlets are found by `get_lsl_inlets`.
            with_name (str): If no inlets are provided, only those inlets whose
                name contains this string will be acquired after discovery.
            dejitter (bool): Whether to regularize inter-sample intervals.
                If `True`, any timestamps returned by LSL are replaced by
                evenly-spaced timestamps based on the stream's nominal sampling
                frequency. Cannot be changed after instantiation due to the
                inconsistencies this would introduce in the resulting data.
            chunk_samples (int): Maximum number of samples per chunk pulled
                from the inlet.
            autostart (bool): Whether to start streaming on instantiation.
            kwargs: Additional keyword arguments to default `data.TimeSeries`.

        """
        # resolve LSL stream if necessary
        if inlets is None:
            inlets = get_lsl_inlets(with_name=with_name)
        else:
            # convert to dict if passed as other iterable
            try:
                inlets.keys()
            except AttributeError:
                inlets = {inlet.info().name(): inlet for inlet in inlets}
        self.inlets = inlets

        # acquire inlet parameters
        self.sfreq, self.n_chan, self.ch_names, self.data = {}, {}, {}, {}
        for name, inlet in inlets:
            info = inlet.info()
            self.sfreq[name] = info.nominal_srate()
            self.n_chan[name] = info.channel_count()
            self.ch_names[name] = get_ch_names(info)
            if '' in self.ch_names[name]:
                raise ValueError("Empty channel name(s) in {} stream info"\
                                 .format(name))
            if not len(self.ch_names[name]) == len(set(self.ch_names[name])):
                raise ValueError("Duplicate channel names in {} stream info"\
                                 .format(name))

            # instantiate the `data.TimeSeries` instances
            metadata = {"pipeline": [type(self).__name__]}
            self.data[name] = data.TimeSeries.with_window(self.ch_names[name],
                                                          self.sfreq[name],
                                                          metadata=metadata,
                                                          **kwargs)

            # aliases for `inlet.pull_chunk`
            self._pull_chunk = {
                name: lambda: inlet.pull_chunk(timeout=1.0,
                                               max_samples=chunk_samples)
                for name, inlet in self.inlets
            }

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

        finally:
            # write any remaining samples in `self.data` to file
            self.data.write_to_file()

    def _new_thread(self):
        # break loop in `stream` to cause thread to return
        self._proceed = False
        # create new thread
        self._thread = threading.Thread(target=self._stream)
        self._proceed = True

    def _dejitter_timestamps(self, timestamps):
        """Partial function for more concise call during loop."""
        last_time = self.data.last_sample['time']
        dejittered = dejitter_timestamps(timestamps, sfreq=self.sfreq,
                                         last_time=last_time)
        return dejittered


def get_lsl_inlets(with_name=None):
    """Resolve all available LSL streams and return the corresponding inlets.

    Args:
        with_name (str): Return only inlets whose names contain this string.
            Case-sensitive; e.g. "Muse" might work if "muse" doesn't.

    Returns:
        dict[str, pylsl.StreamInlet]: LSL inlets of resolved streams.
            Keys are the inlet names.
    """
    streams = [stream for stream in lsl.resolve_streams()]
    try:
        inlets = [lsl.StreamInlet(stream) for name, stream in streams.items()]
        inlets = {inlet.info().name(): inlet for inlet in inlets}
        if with_name is not None:
            inlets = {name: inlet for name, inlet in inlets.items()
                      if with_name in name}
    except IndexError:
        raise IOError("No streams resolved by LSL.")
    return inlets


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
