"""Interfacing between Bluetooth Low Energy and Lab Streaming Layer protocols.

Interfacing with devices over Bluetooth Low Energy (BLE) is achieved using the
`Generic Attribute Profile`_ (GATT) standard procedures for data transfer.
Reading and writing of GATT descriptors is provided by the `pygatt`_ module.

All classes streaming data through an LSL outlet should subclass
`BaseStreamer`.

Also includes dummy streamer objects, which do not acquire data over BLE but
pass local data through an LSL outlet, e.g. for testing.

TODO:
    * AttrDict for attribute-like dict access from device PARAMS?

.. _Generic Attribute Profile:
   https://www.bluetooth.com/specifications/gatt/generic-attributes-overview

.. _pygatt:
   https://github.com/peplin/pygatt
"""

from queue import Queue
from struct import error as StructError
import threading
import time
from warnings import warn

import numpy as np
import pygatt
from pygatt.backends.bgapi.exceptions import ExpectedResponseTimeout
import pylsl as lsl

INFO_ARGS = ['type', 'channel_count', 'nominal_srate', 'channel_format']


class BaseStreamer:
    """Base class for streaming data through an LSL outlet.

    Prepares `pylsl.StreamInfo` and `pylsl.StreamOutlet` objects as well as
    data buffers for handling of incoming chunks.

    Subclasses must implement `start` and `stop` methods for stream control.

    TODO:
        * Public access to outlets and stream info?
        * Push chunks, not samples (have to generate intra-chunk timestamps anyway)
    """

    def __init__(self, device, subscriptions=None, time_func=time.time,
                 ch_names=None):
        """Construct a `BaseStreamer` object.

        Args:
            device: A device module in `ble2lsl.devices`.
            time_func (function): Function for generating timestamps.
            subscriptions (Iterable[str]): Types of device data to stream.
                Some subset of `SUBSCRIPTION_NAMES`.
            ch_names (dict[Iterable[str]]): User-defined channel names.
                e.g. `{'EEG': ('Ch1', 'Ch2', 'Ch3', 'Ch4')}`.
        """
        self._device = device
        if subscriptions is None:
            subscriptions = get_default_subscriptions(device)
        self._subscriptions = tuple(subscriptions)
        self._time_func = time_func
        self._user_ch_names = ch_names if ch_names is not None else {}
        self._stream_params = self._device.PARAMS['streams']

        self._chunk_idxs = stream_idxs_zeros(self._subscriptions)
        self._chunks = empty_chunks(self._stream_params,
                                    self._subscriptions)

        # StreamOutlet.push_chunk doesn't like single-sample chunks...
        # but want to keep using push_chunk for intra-chunk timestamps
        # doing this beforehand to avoid a chunk size check for each push
        chunk_size = self._stream_params["chunk_size"]
        self._push_func = {name: (self._push_chunk_as_sample
                                  if chunk_size[name] == 1
                                  else self._push_chunk)
                           for name in self._subscriptions}

    def start(self):
        """Begin streaming through the LSL outlet."""
        raise NotImplementedError()

    def stop(self):
        """Stop/pause streaming through the LSL outlet."""
        raise NotImplementedError()

    def _init_lsl_outlets(self):
        """Call in subclass after acquiring address."""
        source_id = "{}-{}".format(self._device.NAME, self._address)
        self._info = {}
        self._outlets = {}
        for name in self._subscriptions:
            info = {arg: self._stream_params[arg][name] for arg in INFO_ARGS}
            outlet_name = '{}-{}'.format(self._device.NAME, name)
            self._info[name] = lsl.StreamInfo(outlet_name, **info,
                                              source_id=source_id)
            self._add_device_info(name)
            chunk_size = self._stream_params["chunk_size"][name]
            self._outlets[name] = lsl.StreamOutlet(self._info[name],
                                                   chunk_size=chunk_size,
                                                   max_buffered=360)

    def _push_chunk(self, name, timestamp):
        self._outlets[name].push_chunk(self._chunks[name].tolist(),
                                       timestamp)

    def _push_chunk_as_sample(self, name, timestamp):
        self._outlets[name].push_sample(self._chunks[name].tolist()[0],
                                        timestamp)

    def _add_device_info(self, name):
        """Adds device-specific parameters to `info`."""
        desc = self._info[name].desc()
        try:
            desc.append_child_value("manufacturer", self._device.MANUFACTURER)
        except KeyError:
            warn("Manufacturer not specified in device file")

        channels = desc.append_child("channels")
        try:
            ch_names = self._stream_params["ch_names"][name]
            # use user-specified ch_names if available and right no. channels
            if name in self._user_ch_names:
                user_ch_names = self._user_ch_names[name]
                if len(user_ch_names) == len(ch_names):
                    if len(user_ch_names) == len(set(user_ch_names)):
                        ch_names = user_ch_names
                    else:
                        print("Non-unique names in user-defined {} ch_names; "
                              .format(name), "using default ch_names.")
                else:
                    print("Wrong # of channels in user-defined {} ch_names; "
                          .format(name), "using default ch_names.")

            for c, ch_name in enumerate(ch_names):
                unit = self._stream_params["units"][name][c]
                type_ = self._stream_params["type"][name]
                channels.append_child("channel") \
                    .append_child_value("label", ch_name) \
                    .append_child_value("unit", unit) \
                    .append_child_value("type", type_)
        except KeyError:
            raise ValueError("Channel names, units, or types not specified")

    @property
    def subscriptions(self):
        """The names of the subscribed streams."""
        return self._subscriptions


class Streamer(BaseStreamer):
    """Streams data to an LSL outlet from a BLE device.

    TODO:
        * Try built-in LSL features for intra-chunk timestamps (StreamOutlet)
        * initialize_timestamping: should indices be reset to 0 mid-streaming?
    """

    def __init__(self, device, address=None, backend='bgapi', interface=None,
                 autostart=True, scan_timeout=10.5, internal_timestamps=False,
                 **kwargs):
        """Construct a `Streamer` instance for a given device.

        Args:
            device (dict):  A device module in `ble2lsl.devices`.
                For example, `ble2lsl.devices.muse2016`.
                Provides info on BLE characteristics and device metadata.
            address (str): Device MAC address for establishing connection.
                By default, this is acquired automatically using device name.
            backend (str): Which `pygatt` backend to use.
                Allowed values are `'bgapi'` or `'gatt'`. The `'gatt'` backend
                only works on Linux under the BlueZ protocol stack.
            interface (str): The identifier for the BLE adapter interface.
                When `backend='gatt'`, defaults to `'hci0'`.
            autostart (bool): Whether to start streaming on instantiation.
            scan_timeout (float): Seconds before timeout of BLE adapter scan.
            internal_timestamps (bool): Use internal timestamping.
                If `False` (default), uses initial timestamp, nominal sample
                rate, and device-provided sample ID to determine timestamp.
                If `True` (or when sample IDs not provided), generates
                timestamps at the time of chunk retrieval, only using
                nominal sample rate as need to determine timestamps within
                chunks.
        """
        BaseStreamer.__init__(self, device=device, **kwargs)
        self._transmit_queue = Queue()
        self._ble_params = self._device.PARAMS["ble"]
        self._address = address

        # use internal timestamps if requested, or if stream is variable rate
        # (LSL uses nominal_srate=0.0 for variable rates)
        nominal_srates = self._stream_params["nominal_srate"]
        self._internal_timestamps = {name: (internal_timestamps
                                            if nominal_srates[name] else True)
                                     for name in device.STREAMS}
        self._start_time = stream_idxs_zeros(self._subscriptions)
        self._first_chunk_idxs = stream_idxs_zeros(self._subscriptions)

        # initialize gatt adapter
        if backend == 'bgapi':
            self._adapter = pygatt.BGAPIBackend(serial_port=interface)
        elif backend in ['gatt', 'bluez']:
            # only works on Linux
            interface = self.interface or 'hci0'
            self._adapter = pygatt.GATTToolBackend(interface)
        else:
            raise(ValueError("Invalid backend specified; use bgapi or gatt."))
        self._backend = backend
        self._scan_timeout = scan_timeout

        self._transmit_thread = threading.Thread(target=self._transmit_chunks)

        if autostart:
            self.connect()
            self.start()

    def _init_timestamp(self, name, chunk_idx):
        """Set the starting timestamp and chunk index for a subscription."""
        self._first_chunk_idxs[name] = chunk_idx
        self._start_time[name] = self._time_func()

    def start(self):
        """Start streaming by writing to the send characteristic."""
        self._transmit_thread.start()
        self._ble_device.char_write(self._ble_params['send'],
                                    value=self._ble_params['stream_on'],
                                    wait_for_response=False)

    def stop(self):
        """Stop streaming by writing to the send characteristic."""
        self._ble_device.char_write(self._ble_params["send"],
                                    value=self._ble_params["stream_off"],
                                    wait_for_response=False)

    def send_command(self, value):
        """Write some value to the send characteristic."""
        self._ble_device.char_write(self._ble_params["send"],
                                    value=value,
                                    wait_for_response=False)

    def disconnect(self):
        """Disconnect from the BLE device and stop the adapter.

        Note:
            After disconnection, `start` will not resume streaming.

        TODO:
            * enable device reconnect with `connect`
        """
        self.stop()  # stream_off command
        self._ble_device.disconnect()  # BLE disconnect
        self._adapter.stop()

    def connect(self):
        """Establish connection to BLE device (prior to `start`).

        Starts the `pygatt` adapter, resolves the device address if necessary,
        connects to the device, and subscribes to the channels specified in the
        device parameters.
        """
        adapter_started = False
        while not adapter_started:
            try:
                self._adapter.start()
                adapter_started = True
            except (ExpectedResponseTimeout, StructError):
                continue

        if self._address is None:
            # get the device address if none was provided
            self._address = self._resolve_address(self._device.NAME)
        try:
            self._ble_device = self._adapter.connect(self._address,
                address_type=self._ble_params['address_type'],
                interval_min=self._ble_params['interval_min'],
                interval_max=self._ble_params['interval_max'])

        except pygatt.exceptions.NotConnectedError:
            e_msg = "Unable to connect to device at address {}" \
                .format(self._address)
            raise(IOError(e_msg))

        # initialize LSL outlets and packet handler
        self._init_lsl_outlets()
        self._packet_handler = self._device.PacketHandler(self)

        # subscribe to receive characteristic notifications
        process_packet = self._packet_handler.process_packet
        for name in self._subscriptions:
            try:
                uuids = [self._ble_params[name] + '']
            except TypeError:
                uuids = self._ble_params[name]
            for uuid in uuids:
                if uuid:
                    self._ble_device.subscribe(uuid, callback=process_packet)
            # subscribe to recieve simblee command from ganglion doc

    def _resolve_address(self, name):
        list_devices = self._adapter.scan(timeout=self._scan_timeout)
        for device in list_devices:
            if name in device['name']:
                return device['address']
        raise(ValueError("No devices found with name `{}`".format(name)))

    def _transmit_chunks(self):
        """TODO: missing chunk vs. missing sample"""
        # nominal duration of chunks for progressing non-internal timestamps
        chunk_period = {name: (self._stream_params["chunk_size"][name]
                               / self._stream_params["nominal_srate"][name])
                        for name in self._subscriptions
                        if not self._internal_timestamps[name]}
        first_idx = self._first_chunk_idxs
        while True:
            name, chunk_idx, chunk = self._transmit_queue.get()
            self._chunks[name][:, :] = chunk

            # update chunk index records and report missing chunks
            # passing chunk_idx=-1 to the queue averts this (ex. status stream)
            if not chunk_idx == -1:
                if self._chunk_idxs[name] == 0:
                    self._init_timestamp(name, chunk_idx)
                    self._chunk_idxs[name] = chunk_idx - 1
                if not chunk_idx == self._chunk_idxs[name] + 1:
                    print("Missing {} chunk {}: {}"
                          .format(name, chunk_idx, self._chunk_idxs[name]))
                self._chunk_idxs[name] = chunk_idx
            else:
                # track number of received chunks for non-indexed streams
                self._chunk_idxs[name] += 1

            # generate timestamp; either internally or
            if self._internal_timestamps[name]:
                timestamp = self._time_func()
            else:
                timestamp = chunk_period[name] * (chunk_idx - first_idx[name])
                timestamp += self._start_time[name]

            self._push_func[name](name, timestamp)

    @property
    def backend(self):
        """The name of the `pygatt` backend used by the instance."""
        return self._backend

    @property
    def address(self):
        """The MAC address of the device."""
        return self._address


class Dummy(BaseStreamer):
    """Mimicks a device and pushes local data into an LSL outlet.

    TODO:
        * verify timestamps/delays (seems too fast in plot.Lines)
    """

    def __init__(self, device, chunk_iterator=None, subscriptions=None,
                 autostart=True, **kwargs):
        """Construct a `Dummy` instance.

        Args:
            device: BLE device to impersonate (i.e. from `ble2lsl.devices`).
            chunk_iterator (generator): Class that iterates through chunks.
            autostart (bool): Whether to start streaming on instantiation.
        """
        nominal_srate = device.PARAMS["streams"]["nominal_srate"]
        if subscriptions is None:
            subscriptions = get_default_subscriptions(device)
        subscriptions = {name for name in subscriptions
                         if nominal_srate[name] > 0}

        BaseStreamer.__init__(self, device=device, subscriptions=subscriptions,
                              **kwargs)

        self._address = "DUMMY"
        self._init_lsl_outlets()

        chunk_shapes = {name: self._chunks[name].shape
                        for name in self._subscriptions}
        self._delays = {name: 1 / (nominal_srate[name] / chunk_shapes[name][1])
                        for name in self._subscriptions}

        # generate or load fake data
        if chunk_iterator is None:
            chunk_iterator = NoisySinusoids
        self._chunk_iter = {name: chunk_iterator(chunk_shapes[name],
                                                 nominal_srate[name])
                            for name in self._subscriptions}

        # threads to mimic incoming BLE data
        self._threads = {name: threading.Thread(target=self._stream,
                                                kwargs=dict(name=name))
                         for name in self._subscriptions}

        if autostart:
            self.start()

    def start(self):
        """Start pushing data into the LSL outlet."""
        self._proceed = True
        for name in self._subscriptions:
            self._threads[name].start()

    def stop(self):
        """Stop pushing data. Ends execution of chunk streaming threads.

        Restart requires a new `Dummy` instance.
        """
        self._proceed = False

    def _stream(self, name):
        """Run in thread to mimic periodic hardware input."""
        for chunk in self._chunk_iter[name]:
            if not self._proceed:
                break
            self._chunks[name] = chunk
            timestamp = time.time()
            self._push_func[name](name, timestamp)
            time.sleep(self._delays[name])

    def make_chunk(self, chunk_ind):
        """Prepare a chunk from the totality of local data.

        TODO:
            * replaced when using an iterator
        """
        self._chunks
        # TODO: more realistic timestamps
        timestamp = self._time_func()
        self._timestamps = np.array([timestamp]*self._chunk_size)


def stream_idxs_zeros(subscriptions):
    """Initialize an integer index for each subscription."""
    idxs = {name: 0 for name in subscriptions}
    return idxs


def empty_chunks(stream_params, subscriptions):
    """Initialize an empty chunk array for each subscription."""
    chunks = {name: np.zeros((stream_params["chunk_size"][name],
                              stream_params["channel_count"][name]),
                             dtype=stream_params["numpy_dtype"][name])
              for name in subscriptions}
    return chunks


def get_default_subscriptions(device):
    # look for default list; if unavailable, subscribe to all
    try:
        subscriptions = device.DEFAULT_SUBSCRIPTIONS
    except AttributeError:
        subscriptions = device.STREAMS
    return subscriptions


class ChunkIterator:
    """Generator object (i.e. iterator) that yields chunks.

    Placeholder until I figure out how this might work as a base class.
    """

    def __init__(self, chunk_shape, srate):
        self._chunk_shape = chunk_shape
        self._srate = srate


class NoisySinusoids(ChunkIterator):
    """Iterator class to provide noisy sinusoidal chunks of data."""

    def __init__(self, chunk_shape, srate, freqs=[5, 10, 12, 20], noise_std=1):
        super().__init__(chunk_shape=chunk_shape, srate=srate)
        self._ang_freqs = 2 * np.pi * np.array(freqs)
        self._speriod = 1 / self._srate
        self._chunk_t_incr = (1 + chunk_shape[0]) / self._srate
        self._freq_amps = np.random.randint(1, 5, len(freqs))
        self._noise_std = noise_std

    def __iter__(self):
        self._t = (np.arange(self._chunk_shape[0]).reshape((-1, 1))
                   * self._speriod)
        return self

    def __next__(self):
        # start with noise
        chunk = np.random.normal(0, self._noise_std, self._chunk_shape)

        # sum frequencies with random amplitudes
        for i, freq in enumerate(self._ang_freqs):
            chunk += self._freq_amps[i] * np.sin(freq * self._t)

        self._t += self._chunk_t_incr

        return chunk
