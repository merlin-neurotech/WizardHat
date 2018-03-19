"""Interfacing between Bluetooth Low Energy and Lab Streaming Layer protocols.

Interfacing with devices over Bluetooth Low Energy (BLE) is achieved using the
`Generic Attribute Profile`_ (GATT) standard procedures for data transfer.
Reading and writing of GATT descriptors is provided by the `pygatt`_ module.

All classes streaming data through an LSL outlet should subclass
`OutletStreamer`.

Also includes dummy streamer objects, which do not acquire data over BLE but
pass local data through an LSL outlet, e.g. for testing.

TODO:
    * Documentation for both BLE backends.

.. _Generic Attribute Profile:
   https://www.bluetooth.com/specifications/gatt/generic-attributes-overview

.. _pygatt:
   https://github.com/peplin/pygatt
"""

import time

import bitstring
import numpy as np
import pygatt
import pylsl as lsl
import threading


class OutletStreamer:
    """Base class for streaming data through an LSL outlet.

    Prepares `pylsl.StreamInfo` and `pylsl.StreamOutlet` objects as well as
    data buffers for handling of incoming chunks.

    Subclasses must implement `start` and `stop` methods for stream control.

    Attributes:
        info (pylsl.StreamInfo): Contains information about the stream.
        outlet (pylsl.StreamOutlet): LSL outlet to which data is pushed.

    TODO:
        * get source_id... serial number from Muse? MAC address?
        * Implement with abc.ABC
        * Some way to generalize autostart behaviour?
    """

    def __init__(self, device_params=None, stream_params=None, chunk_size=12,
                 time_func=time.time):
        """Construct an `OutletStreamer` object.

        Args:
            stream_params (dict): Parameters to construct `pylsl.StreamInfo`.
            chunk_size (int): Number of samples pushed per LSL chunk.
            time_func (function): Function for generating timestamps.
        """
        if device_params is None:
            device_params = {}
        self._device_params = device_params
        if stream_params is None:
            stream_params = {}
        self._stream_params = stream_params
        self._chunk_size = chunk_size
        self._time_func = time_func

        # construct LSL StreamInfo and StreamOutlet
        self.info = lsl.StreamInfo(**stream_params, source_id='MuseNone')
        self.outlet = lsl.StreamOutlet(self.info, chunk_size=chunk_size,
                                       max_buffered=360)

        self._init_sample()

    def start(self):
        """Begin streaming through the LSL outlet."""
        raise NotImplementedError()

    def stop(self):
        """Stop/pause streaming through the LSL outlet."""
        raise NotImplementedError()

    def _init_sample(self):
        self._timestamps = np.zeros(self.info.channel_count())
        self._data = np.zeros((self.info.channel_count(), self._chunk_size))

    def _push_chunk(self, channels, timestamps):
        for sample in range(self._chunk_size):
            self.outlet.push_sample(channels[:, sample], timestamps[sample])

    def _add_device_info(self):
        """Adds device-specific parameters to `info`."""
        self.info.desc().append_child_value("manufacturer",
                                            self._device_params["manufacturer"])

        self.channels = self.info.desc().append_child("channels")
        for ch_name in self._device_params["ch_names"]:
            self.channels.append_child("channel") \
                .append_child_value("label", ch_name) \
                .append_child_value("unit", self._device_params["units"]) \
                .append_child_value("type", self._stream_params["type"])


class BLEStreamer(OutletStreamer):
    """Streams data to an LSL outlet from a BLE device.

    Attributes:
        address (str): Device-specific (MAC) address.
        start_time (float): Time of timestamp initialization by `initialize`.
            Provided by `time_func`.

    TODO:
        * device-dependent bit indices in _transmit_packet
        * device-dependent unit conversion in _unpack_channel
        * move fake data generator to outside/iterator?
    """

    def __init__(self, device_params, address=None, backend='bgapi',
                 interface=None, autostart=True, **kwargs):
        """Construct a `BLEStreamer` instance.

        Args:
            device_params (dict): Device-specific parameters.
                Includes GATT handles and characteristics, channel UUIDs and
                names, and other device metadata.
            address (str): Device MAC address for establishing connection.
                By default, this is acquired automatically.
            backend (str): Which `pygatt` backend to use.
                Allowed values are `'bgapi'` or `'gatt'`. The `'gatt'` backend
                only works on Linux.
            interface (str): The identifier for the BLE adapter interface.
                When `backend='gatt'`, defaults to `'hci0'`.
            autostart (bool): Whether to start streaming on instantiation.
        """
        OutletStreamer.__init__(self, device_params=device_params, **kwargs)
        self.address = address

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

        self._set_packet_format()
        self._add_device_info()
        self._ble_params = self._device_params["ble"]

        self.initialize_timestamping()

        if autostart:
            self.connect()
            self.start()

    @classmethod
    def from_device(cls, device, **kwargs):
        """Construct a `DeviceStreamer` from a device in `ble2lsl.devices`.

        Args:
           device: A device module in `ble2lsl.devices`.
               For example, `ble2lsl.devices.muse2016`.
        """
        return cls(device_params=device.PARAMS,
                   stream_params=device.STREAM_PARAMS,
                   **kwargs)

    def initialize_timestamping(self):
        """Reset the parameters for timestamp generation."""
        self._sample_index = 0
        self._last_tm = 0
        self.start_time = self._time_func()

    def start(self):
        """Start streaming by writing to the relevant GATT characteristic."""
        self._device.char_write_handle(self._ble_params["handle"],
                                       value=self._ble_params["stream_on"],
                                       wait_for_response=False)

    def stop(self):
        """Stop streaming by writing to the relevant GATT characteristic."""
        self._device.char_write_handle(self._ble_params["handle"],
                                       value=self._ble_params["stream_off"],
                                       wait_for_response=False)

    def disconnect(self):
        """Disconnect from the BLE device and stop the adapter.

        Note:
            After disconnection, `start` will not resume streaming.

        TODO:
            * enable device reconnect with `connect`
        """
        self.stop()
        self._device.disconnect()
        self._adapter.stop()

    def connect(self):
        """Establish connection to BLE device (prior to `start`).

        Starts the `pygatt` adapter, resolves the device address if necessary,
        connects to the device, and subscribes to the channels specified in the
        device parameters.
        """
        self._adapter.start()
        if self.address is None:
            # get the device address if none was provided
            self.address = self._resolve_address(self._stream_params["name"])
        try:
            # connect to the device
            self._device = self._adapter.connect(self.address)
        except pygatt.exceptions.NotConnectedError:
            e_msg = "Unable to connect to device at address {}" \
                    .format(self.address)
            raise(IOError(e_msg))

        # subscribe to the channels specified by the device parameters
        for uuid in self._device_params["ch_uuids"]:
            self._device.subscribe(uuid, callback=self._transmit_packet)

    def _resolve_address(self, name):
        list_devices = self._adapter.scan(timeout=10.5)
        for device in list_devices:
            if name in device['name']:
                return device['address']
        raise(ValueError("No devices found with name `{}`".format(name)))

    def _set_packet_format(self):
        """Defines the format for parsing bit strings received over BLE."""
        dtypes = self._device_params["packet_dtypes"]
        self._packet_format = dtypes["index"] + \
                              (',' + dtypes["ch_value"]) * self._chunk_size

    def _unpack_channel(self, packet):
        """Parse the bitstrings received over BLE."""
        packet_bits = bitstring.Bits(bytes=packet)
        unpacked = packet_bits.unpack(self._packet_format)

        packet_index = unpacked[0]
        packet_values = np.array(unpacked[1:])
        packet_values = 0.48828125 * (packet_values - 2048)

        return packet_index, packet_values

    def _transmit_packet(self, handle, data):
        """Callback function used by `pygatt` to receive BLE data."""
        timestamp = self._time_func()
        index = int((handle - 32) / 3)

        tm, d = self._unpack_channel(data)

        if self._last_tm == 0:
            self._last_tm = tm - 1

        self._data[index] = d
        self._timestamps[index] = timestamp

        # if last channel in chunk
        if handle == 35:
            if tm != self._last_tm + 1:
                print("Missing sample {} : {}".format(tm, self._last_tm))
            self._last_tm = tm

            # sample indices
            sample_indices = np.arange(self._chunk_size) + self._sample_index
            self._sample_index += self._chunk_size

            timestamps = sample_indices / self.info.nominal_srate() \
                         + self.start_time

            self._push_chunk(self._data, timestamps)
            self._init_sample()

    @property
    def backend(self):
        """The `pygatt` backend used by the instance."""
        return self._backend


class DummyStreamer(OutletStreamer):
    """Streams data over an LSL outlet from a local source.

    Attributes:
        sfreq (int): Sampling frequency for generated data.
        n_chan (int): Number of channels of generated data.
        csv_file (str): Filename of `.csv` containing local data.

    TODO:
        * take a data iterator instead of multiple optional parameters
    """

    def __init__(self, dummy_func=None, dur=60, sfreq=256, n_chan=4,
                 csv_file=None, autostart=True, **kwargs):
        """Construct a `DummyStreamer` instance.

        Attributes:

        """
        if 'device_params' not in kwargs:
            device_params = {'ch_names': [''] * n_chan}

        if 'stream_params' not in kwargs:
            kwargs['stream_params'] = {'name': 'Dummy',
                                       'type': 'EEG',
                                       'channel_count': n_chan,
                                       'nominal_srate': sfreq,
                                       'channel_format': 'float32'}

        OutletStreamer.__init__(self, **kwargs)

        self._thread = threading.Thread(target=self._stream)

        # generate or load fake data
        if csv_file is None:
            self.sfreq = sfreq
            self.n_chan = n_chan
            self._dummy_data = self.gen_dummy_data(dur)
        else:
            self.csv_file = csv_file
            # TODO:load csv file to np array

        self._proceed = True
        if autostart:
            self._thread.start()

    @classmethod
    def impersonate_device(cls, device, **kwargs):
        """Construct a `DummyStreamer` instance to mimic a device's metadata.

        Args:
            device: A device module from `ble2lsl.devices`.
        """
        return cls(sfreq=device.STREAM_PARAMS['nominal_srate'],
                   n_chan=device.STREAM_PARAMS['channel_count'],
                   device_params=device.PARAMS,
                   stream_params=device.STREAM_PARAMS,
                   **kwargs)

    def _stream(self):
        while self._proceed:
            self.start_time = self._time_func()
            chunk_inds = np.arange(0, len(self._dummy_data.T), self._chunk_size)
            for chunk_ind in chunk_inds:
                self.make_chunk(chunk_ind)
                self._push_chunk(self._data, self._timestamps)
                # force sampling rate
                sec_per_chunk = 1 / (self.sfreq / self._chunk_size)
                time.sleep(sec_per_chunk)

    def gen_dummy_data(self, dur, freqs=[5, 10, 12, 20]):
        """Generate noisy sinusoidal dummy samples.

        TODO:
            * becomes external when passing an iterator to `DummyStreamer`
        """
        n_samples = dur * self.sfreq
        x = np.arange(0, n_samples)
        a_freqs = 2 * np.pi * np.array(freqs)
        y = np.zeros((self.n_chan, len(x)))

        # sum frequencies with random amplitudes
        for freq in a_freqs:
            y += np.random.randint(1,5) * np.sin(freq * x)

        noise = np.random.normal(0, 1, (self.n_chan, n_samples))
        dummy_data = y + noise

        return dummy_data

    def make_chunk(self, chunk_ind):
        """Prepare a chunk from the totality of local data.

        TODO:
            * replaced when using an iterator
        """
        self._data = self._dummy_data[:,chunk_ind:chunk_ind+self._chunk_size]
        # TODO: more realistic timestamps
        timestamp = self._time_func()
        self._timestamps = np.array([timestamp]*self._chunk_size)
