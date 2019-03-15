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
    * auto-acquire device if none given?

.. _Lab Streaming Layer:
   https://github.com/sccn/labstreaminglayer
"""

from wizardhat.buffers import TimeSeries

from serial.serialutil import SerialException
import threading
import warnings

import numpy as np
import pylsl as lsl


class Receiver:
    """Receives data from one or more LSL streams associated with a device.

    Attributes:
        inlets (dict[pylsl.StreamInlet]): The LSL inlet(s) to acquire.
        data (buffers.TimeSeries): Object in which the incoming data is stored,
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

    def __init__(self, source_id=None, with_types=('',), dejitter=True,
                 max_chunklen=0, autostart=True, window=10, **kwargs):
        """Instantiate LSLStreamer given length of data store in seconds.

        Args:
            source_id (str): Full or partial source ID for the streamed device.
            with_types (Iterable[str]): Only
                matching one of these types will be acquired. For example,
                `with_types=('EEG', 'accelerometer')`.
            dejitter (bool): Whether to regularize inter-sample intervals.
                If `True`, any timestamps returned by LSL are replaced by
                evenly-spaced timestamps based on the stream's nominal sampling
                frequency. Cannot be changed after instantiation due to the
                inconsistencies this would introduce in the resulting data.
            max_chunklen (int): Maximum number of samples per chunk pulled
                from the inlets. Default: 0 (determined at stream outlet).
            autostart (bool): Whether to start streaming on instantiation.
            kwargs: Additional keyword arguments to default `buffers.TimeSeries`.

        """

        streams = get_lsl_streams()
        source_ids = list(sorted(streams.keys()))

        if source_id is not None:
            matches = [(source_id in sid) for sid in source_ids]
            if not any(matches):
                raise ValueError("No source found matching ID: {}"
                                 .format(source_id))
            menu_intro = ("Multiple sources match given ID ({})."
                          .format(source_id))
        else:
            matches = source_ids
            menu_intro = "Multiple sources detected."

        # let user choose from a menu if multiple sources are detected
        # or if user-defined `source_id` partially matches more than one
        if len(matches) > 1:
            source_id = utils.menu_numlist(matches, menu_intro=menu_intro)
        else:
            try:
                source_id = source_ids[0]
            except IndexError:
                # TODO: more specific error
                raise RuntimeError("No LSL streams detected")
        print("Using source with ID {}".format(source_id))
        self._source_id = source_id

        self._inlets = get_lsl_inlets(streams,
                                      with_types=with_types,
                                      with_source_ids=(source_id,),
                                      max_chunklen=max_chunklen)[source_id]

        # acquire inlet parameters
        self.buffers = {}
        self.sfreq, self.n_chan, self.ch_names = {}, {}, {}
        for name, inlet in list(self._inlets.items()):
            info = inlet.info()
            self.sfreq[name] = info.nominal_srate()
            # TODO: include message/status streams?
            if self.sfreq[name] < 1 / window:
                warn_msg = ("Stream '{}' sampling period larger".format(name)
                            + " than buffer window: will not be stored")
                print(warn_msg)
                self._inlets.pop(name)
                continue
            self.n_chan[name] = info.channel_count()
            self.ch_names[name] = get_ch_names(info)
            if '' in self.ch_names[name]:
                print("Empty channel name(s) in {} stream info"
                      .format(name))
            if not len(self.ch_names[name]) == len(set(self.ch_names[name])):
                print("Duplicate channel names in {} stream info"
                      .format(name))

            # instantiate the `buffers.TimeSeries` instances
            metadata = {"pipeline": [type(self).__name__]}
            self.buffers[name] = TimeSeries.with_window(self.ch_names[name],
                                                        self.sfreq[name],
                                                        metadata=metadata,
                                                        label=info.name(),
                                                        window=window,
                                                        **kwargs)

        self._dejitter = dejitter
        self._threads = {}
        self._new_threads()
        if autostart:
            self.start()

    def __getitem__(self, name):
        """Allows instances to be indexed for their buffers by stream name.

        Example:
            >>> assert my_receiver[name] is my_receiver.buffers[name]
        """
        return self.buffers[name]

    @classmethod
    def record(cls, duration, **kwargs):
        """Collect data over a finite interval, then stop."""
        return cls(window=duration, store_once=True, **kwargs)

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
            for name in self._inlets:
                self._threads[name].start()
        except RuntimeError:
            for name in self._inlets:
                if self._thread.ident:
                    # thread exists but has stopped; create and start new one
                    self._new_threads([name])
                    self._threads[name].start()
                else:
                    print("Streaming has already started!")

    def stop(self):
        """Stop data streaming."""
        self._proceed = False

    def _receive(self, name):
        """Streaming thread."""
        inlets = self._inlets
        try:
            while self._proceed[name]:
                samples, timestamps = inlets[name].pull_chunk(timeout=0.0)

                if timestamps:
                    if self._dejitter:
                        try:
                            timestamps = self._dejitter_timestamps(name,
                                                                   timestamps)
                        except IndexError:
                            print(name)
                    self.buffers[name].update(timestamps, samples)

        except SerialException:
            print("BGAPI streaming interrupted. Device disconnected?")

        finally:
            # write any remaining samples in `self.buffers` to file
            self.buffers[name].write_to_file()

    def _new_threads(self, names=None):
        # break loop in `stream` to cause thread to return
        if names is None:
            names = self._inlets.keys()
        self._proceed = dict(zip(names, [False] * len(names)))
        # create new threads
        for name in names:
            self._threads[name] = threading.Thread(target=self._receive,
                                                   kwargs=dict(name=name))
        self._proceed = dict(zip(names, [True] * len(names)))

    def _dejitter_timestamps(self, name, timestamps):
        """Partial function for more concise call during loop."""
        last_time = self.buffers[name].last_sample['time']
        if self.sfreq[name] > 0:
            dejittered = dejitter_timestamps(timestamps,
                                             sfreq=self.sfreq[name],
                                             last_time=last_time)
        else:
            dejittered = timestamps
        return dejittered


def get_lsl_streams():
    """Discover all LSL streams available on the local network.

    Returns:
        dict[str, dict[str, pylsl.StreamInfo]]: Streams mapped to source/type.
            Keys are source IDs; values are dictionaries for which the keys
            are stream types and the values are stream.

    Example:
        When EEG and accelerometer streams are found for a single Muse headset:

        >>> get_lsl_streams()
        {'Muse-00:00:00:00:00:00': {'EEG': <pylsl.pylsl.StreamInfo>,
                                    'accelerometer': <pylsl.pylsl.StreamInfo>}}
    """
    streams = [(stream.source_id(), stream.type(), stream)
               for stream in lsl.resolve_streams(wait_time=2)]
    streams_dict = streams_dict_from_streams(streams)
    return streams_dict


def streams_dict_from_streams(streams):
    """Convert a list of stream info objects into a source/type mapping.

    Args:
        streams (Iterable[pylsl.StreamInfo]): List of stream info objects,
            typically as returned by `pylsl.resolve_streams()`.

    Returns:
        dict[str, dict[str, pylsl.StreamInfo]]: Streams mapped to source/type.
            Keys are source IDs; values are dictionaries for which the keys
            are stream types and the values are stream.
    """
    source_ids = set(stream[0] for stream in streams)
    streams_dict = dict(zip(source_ids, [{} for _ in source_ids]))
    for source_id, stream_type, stream_info in streams:
        streams_dict[source_id][stream_type] = stream_info
    return streams_dict


def get_source_ids():
    """Convenience function to list available LSL sources (i.e. devices)."""
    source_ids = tuple(get_lsl_streams().keys())
    return source_ids


def get_lsl_inlets(streams=None, with_source_ids=('',), with_types=('',),
                   max_chunklen=0):
    """Return LSL stream inlets for given/discovered LSL streams.

    If `streams` is not given, will automatically discover all available
    streams.

    Args:
        streams: List of `pylsl.StreamInfo` or source/type mapping.
            See `streams_dict_from_streams` for additional documentation
            of the difference between the two data types.
        with_source_id (Iterable[str]): Return only inlets whose source ID
            contains one of these strings.
            Case-sensitive; e.g. "Muse" might work if "muse" doesn't.
        with_type (Iterable[str]): Return only inlets with these stream types.

    Returns:
        dict[str, dict[str, pylsl.StreamInlet]]: LSL inlet objects.
            Keys are the source IDs; values are dicts where the keys are stream
            types and values are stream inlets.

    TODO:
        * Try leveraging lsl.resolve_byprop or lsl.resolve_bypred
        * inlet time_correction necessary for remotely generated timestamps?
    """
    if streams is None:
        streams = get_lsl_streams()
    else:
        # ensure streams is in streams_dict format
        try:  # quack
            streams.keys()
            list(streams.values())[0].keys()
        except AttributeError:
            streams = streams_dict_from_streams(streams)
        except IndexError:
            pass
    streams_dict = streams

    inlets = dict(zip(streams_dict.keys(), [{} for _ in streams_dict]))
    for source_id, streams in streams_dict.items():
        if any(id_str in source_id for id_str in with_source_ids):
            for stream_type, stream in streams.items():
                if any(type_str in stream_type for type_str in with_types):
                    inlets[source_id][stream_type] = lsl.StreamInlet(stream)

    # make sure no empty devices are included following inclusion rules
    inlets = {source_id: inlets for source_id, inlets in inlets.items()
              if not inlets == {}}

    if inlets == {}:
        print("No inlets created based on the available streams/given rules")

    return inlets


def get_ch_names(info):
    """Return the channel names associated with an LSL stream.

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
        last_time = -1 / sfreq
    dejittered = np.arange(len(timestamps), dtype=np.float64)
    dejittered /= sfreq
    dejittered += last_time + 1 / sfreq
    return dejittered
