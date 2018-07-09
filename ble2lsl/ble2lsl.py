"""Interfacing between Bluetooth Low Energy and Lab Streaming Layer protocols.

Interfacing with devices over Bluetooth Low Energy (BLE) is achieved using the
`Generic Attribute Profile`_ (GATT) standard procedures for data transfer.
Reading and writing of GATT descriptors is provided by the `pygatt`_ module.

All classes streaming data through an LSL outlet should subclass
`BaseStreamer`.

Also includes dummy streamer objects, which do not acquire data over BLE but
pass local data through an LSL outlet, e.g. for testing.

TODO:
    * Documentation for both BLE backends.

.. _Generic Attribute Profile:
   https://www.bluetooth.com/specifications/gatt/generic-attributes-overview

.. _pygatt:
   https://github.com/peplin/pygatt
"""

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

    Attributes:
        info (pylsl.StreamInfo): Contains information about the stream.
        outlet (pylsl.StreamOutlet): LSL outlet to which data is pushed.
    """

    def __init__(self, device, subscriptions=('channels',),
                 time_func=time.time):
        """Construct a `BaseStreamer` object.

        Args:
            device: A device module in `ble2lsl.devices`.
            time_func (function): Function for generating timestamps.
            subscriptions (Iterable[str]): Types of device data to stream.
                Some subset of `SUBSCRIPTION_NAMES`.
        """
        self._device = device
        self._subscriptions = subscriptions
        self._time_func = time_func
        self._stream_params = self._device.PARAMS['streams']

        self._init_buffer()

    def start(self):
        """Begin streaming through the LSL outlet."""
        raise NotImplementedError()

    def stop(self):
        """Stop/pause streaming through the LSL outlet."""
        raise NotImplementedError()

    def _init_lsl_outlets(self):
        """Call in subclass after acquiring address."""
        source_id = "{}-{}".format(self._device.PARAMS['name'], self._address)
        self._info = {}
        self._outlets = {}
        for name in self._subscriptions:
            info = {arg: self._stream_params[arg][name] for arg in INFO_ARGS}
            name = '{}-{}'.format(self._device.PARAMS['name'], name)
            self._info[name] = lsl.StreamInfo(name, **info,
                                              source_id=source_id)
            self._add_device_info(name)
            chunk_size = self._stream_params["chunk_sizes"][name]
            self._outlets[name] = lsl.StreamOutlet(self._info[name],
                                                   chunk_size=chunk_size,
                                                   max_buffered=360)

    def _init_buffer(self):
        channel_counts = self._stream_params["channel_count"]
        chunk_sizes = self._stream_params["chunk_size"]
        dtypes = self._stream_params["numpy_dtypes"]
        self._chunk_timestamps = {name: np.zeros(channel_counts[name],
                                                 dtype=np.float32)
                                  for name in self._subscriptions}
        self._current_chunks = {name: np.zeros((channel_counts[name],
                                                chunk_sizes[name]),
                                               dtype=dtypes[name])
                                for name in self._subscriptions}

    def _push_chunk(self, name):
        for sample in range(self._stream_params["chunk_size"][name]):
            self.outlet.push_sample(self._current_chunks[name][:, sample],
                                    self._chunk_timestamps[name][sample])

    def _add_device_info(self, name):
        """Adds device-specific parameters to `info`."""
        desc = self._info[name].desc()
        try:
            desc.append_child_value("manufacturer",
                                    self._device.PARAMS["manufacturer"])
        except KeyError:
            warn("Manufacturer not specified in device PARAMS")

        channels = desc.append_child("channels")
        try:
            for c, ch_name in enumerate(self._stream_params["ch_names"][name]):
                unit = self._stream_params["units"][name][c]
                type_ = self._stream_params["type"][name]
                channels.append_child("channel") \
                    .append_child_value("label", ch_name) \
                    .append_child_value("unit", unit) \
                    .append_child_value("type", type_)
        except KeyError:
            raise ValueError("Channel names, units, or types not specified")


class Streamer(BaseStreamer):
    """Streams data to an LSL outlet from a BLE device.

    Attributes:
        address (str): Device-specific (MAC) address.
        start_time (float): Time of timestamp initialization by `initialize`.
            Provided by `time_func`.
    """

    def __init__(self, device, subscriptions, address=None, backend='bgapi',
                 interface=None, autostart=True, scan_timeout=10.5,
                 internal_timestamps=False, **kwargs):
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
        self._packet_handler = device.PacketHandler(self._transmit_callback,
                                                    subscriptions)
        self._ble_params = self._device.PARAMS["ble"]
        self._address = address

        # initialize gatt adapter
        if backend == 'bgapi':
            self._adapter = pygatt.BGAPIBackend(serial_port=interface)
        elif backend == 'gatt':
            # only works on Linux
            interface = self.interface or 'hci0'
            self._adapter = pygatt.GATTToolBackend(interface)
        else:
            raise(ValueError("Invalid backend specified; use bgapi or gatt."))
        self._backend = backend
        self._scan_timeout = scan_timeout

        self.initialize_timestamping()
        # self._update_thread = threading.Thread(target=self._transmit_samples)

        if autostart:
            self.connect()
            self.start()

    def initialize_timestamping(self):
        """Reset the parameters for timestamp generation."""
        self._sample_idx = {name: 0 for name in self._subscriptions}
        self._last_idx = {name: 0 for name in self._subscriptions}
        self.start_time = self._time_func()

    def start(self):
        """Start streaming by writing to the relevant GATT characteristic."""
        # self._update_thread.start()
        self._ble_device.char_write(self._ble_params['send'],
                                    value=self._ble_params['stream_on'],
                                    wait_for_response=False)

    def stop(self):
        """Stop streaming by writing to the relevant GATT characteristic."""
        self._ble_device.char_write(self._ble_params["send"],
                                    value=self._ble_params["stream_off"],
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
            except ExpectedResponseTimeout:
                continue

        if self._address is None:
            # get the device address if none was provided
            self._address = self._resolve_address(self._lsl_info["name"])
            try:
                self._ble_device = self._adapter.connect(self._address,
                    address_type=self._ble_params['address_type'],
                    interval_min=self._ble_params['interval_min'],
                    interval_max=self._ble_params['interval_max'])

            except pygatt.exceptions.NotConnectedError:
                e_msg = "Unable to connect to device at address {}" \
                    .format(self._address)
                raise(IOError(e_msg))

        self._init_lsl_outlets()

        # subscribe to receive characteristic notifications
        for name in self._subscriptions:
            try:
                uuids = [self._ble_params[name] + '']
            except TypeError:
                uuids = self._ble_params[name]
            for uuid in uuids:
                self._ble_device.subscribe(uuid,
                                           callback=self._packet_callback)
            # subscribe to recieve simblee command from ganglion doc

    def _resolve_address(self, name):
        list_devices = self._adapter.scan(timeout=self._scan_timeout)
        for device in list_devices:
            if name in device['name']:
                return device['address']
        raise(ValueError("No devices found with name `{}`".format(name)))

    def _packet_callback(self, handle, data):
        """Callback function used by `pygatt` to receive BLE data."""
        self._packet_handler.process_packet(handle, data)

    def _transmit_callback(self, name, sample_idxs, chunk):
        """TODO: missing chunk vs. missing sample"""
        if sample_idxs is None:
            pass
        self._current_chunks[name][:, :] = chunk
        chunk_idx = sample_idxs[0]
        if self._last_idx[name] == 0:
            self._last_idx[name] = chunk_idx - 1
        if not chunk_idx == self._last_idx[name] + 1:
            print("Missing sample {} : {}".format(chunk_idx,
                                                  self._last_idx[name]))
        self._last_idx = chunk_idx
        sample_idxs = np.arange(self._chunk_size)
        if not self._internal_timestamping:
            sample_idxs += self._sample_idx[name]
        self._sample_idx[name] += self._device.PARAMS["chunk_size"][name]

        # generate timestamps based on start time and nominal sample rate
        timestamps = sample_idxs / self._device.PARAMS["nominal_srate"][name]
        if self._internal_timestamping:
            timestamps += self._time_func()
        else:
            timestamps += self.start_time
        self._chunk_timestamps[name] = timestamps
        self._push_chunk(name)

    @property
    def backend(self):
        """The `pygatt` backend used by the instance."""
        return self._backend

    @property
    def address(self):
        """The MAC address of the device."""
        return self._address

class Dummy(BaseStreamer):
    """Streams data over an LSL outlet from a local source.

    Attributes:
        csv_file (str): Filename of `.csv` containing local data.

    TODO:
        * take a data iterator instead of CSV file/single random generator
        * implement CSV file streaming
    """

    def __init__(self, device, dur=60, csv_file=None, autostart=True,
                 **kwargs):
        """Construct a `Dummy` instance.

        Attributes:
            device: BLE device to impersonate (i.e. from `ble2lsl.devices`).
            dur (float): Duration of random data to generate and stream.
                The generated data is streamed on a loop.
            csv_file (str): CSV file containing pre-generated data to stream.
            autostart (bool): Whether to start streaming on instantiation.
        """

        BaseStreamer.__init__(self, device=device, **kwargs)

        self._address = None
        self._init_lsl_outlet()
        self._thread = threading.Thread(target=self._stream)

        # generate or load fake data
        if csv_file is None:
            self._sfreq = device.LSL_INFO['nominal_srate']
            self._n_chan = device.LSL_INFO['channel_count']
            self._dummy_data = self.gen_dummy_data(dur)
        else:
            self.csv_file = csv_file
            # TODO: load csv file to np array

        self._proceed = True
        if autostart:
            self._thread.start()

    def _stream(self):
        """Run in thread to mimic periodic hardware input."""
        sec_per_chunk = 1 / (self._sfreq / self._chunk_size)
        while self._proceed:
            self.start_time = self._time_func()
            chunk_inds = np.arange(0, len(self._dummy_data.T),
                                   self._chunk_size)
            for chunk_ind in chunk_inds:
                self.make_chunk(chunk_ind)
                self._push_chunk(self._current_chunks, self._timestamps)
                # force sampling rate
                time.sleep(sec_per_chunk)

    def gen_dummy_data(self, dur, freqs=[5, 10, 12, 20]):
        """Generate noisy sinusoidal dummy samples.

        TODO:
            * becomes external when passing an iterator to `Dummy`
        """
        n_samples = dur * self._sfreq
        x = np.arange(0, n_samples)
        a_freqs = 2 * np.pi * np.array(freqs)
        y = np.zeros((self._n_chan, len(x)))

        # sum frequencies with random amplitudes
        for freq in a_freqs:
            y += np.random.randint(1, 5) * np.sin(freq * x)

        noise = np.random.normal(0, 1, (self._n_chan, n_samples))
        dummy_data = y + noise

        return dummy_data

    def make_chunk(self, chunk_ind):
        """Prepare a chunk from the totality of local data.

        TODO:
            * replaced when using an iterator
        """
        self._current_chunks = self._dummy_data[:, chunk_ind:chunk_ind+self._chunk_size]
        # TODO: more realistic timestamps
        timestamp = self._time_func()
        self._timestamps = np.array([timestamp]*self._chunk_size)
