"""Interfacing between Bluetooth Low Energy and Lab Streaming Layer protocols.

Interfacing with devices over Bluetooth Low Energy (BLE) is achieved using the
`Generic Attribute Profile`_ (GATT) standard procedures for data transfer.
Reading and writing of GATT descriptors is provided by the `pygatt`_ module.

All classes streaming data through an LSL outlet should subclass
`OutletStreamer`.

Also includes dummy streamer objects, which do not acquire data over BLE but
pass local data through an LSL outlet, e.g. for testing.

TODO:
    * Documentation for the two BLE backends.

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
        info (pylsl.StreamInfo):
        outlet (pylsl.StreamOutlet):

    TODO:
        * get source_id... serial number from Muse? MAC address?
        * Implement with abc.ABC
        * Some way to generalize autostart behaviour?
    """

    def __init__(self, stream_params=None, chunk_size=None,
                 time_func=time.time):
        """Construct an `OutletStreamer` object.

        Args:
            stream_params (dict): Parameters to construct `pylsl.StreamInfo`
            chunk_size (int): Number of samples pushed per LSL chunk
            time_func (function): Function for generating timestamps
        """
        if stream_params is None:
            stream_params = {}
        self._stream_params = stream_params
        self._chunk_size = chunk_size
        self._time_func = time_func

        # construct LSL StreamInfo and StreamOutlet
        self.info = lsl.StreamInfo(**stream_params, source_id='')
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
        for sample in range(self.chunk_size):
            pass
            #self.outlet.push_sample(channels[:, sample], timestamps[sample])


class DeviceStreamer(OutletStreamer):
    """

    """

    def __init__(self, device_params=None, stream_params=None, interface=None,
                 address=None, backend='bgapi', autostart=True, chunk_size=12,
                 **kwargs):
        """

        """
        OutletStreamer.__init__(self, stream_params=stream_params,
                                chunk_size=chunk_size, **kwargs)
        self.device_params = device_params
        self.interface = interface
        self.address = address

        # initialize gatt adapter
        if backend == 'gatt':
            self.interface = self.interface or 'hci0'
            self.adapter = pygatt.GATTToolBackend(self.interface)
        elif backend == 'bgapi':
            self.adapter = pygatt.BGAPIBackend(serial_port=self.interface)
        else:
            raise(ValueError("Invalid backend specified; use bgapi or gatt."))
        self.backend = backend

        self._set_packet_format()
        self._add_device_info()

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

    def connect(self):
        """

        """
        self.adapter.start()
        if self.address is None:
            self.address = self._get_device_address(self.stream_params["name"])
        try:
            self.device = self.adapter.connect(self.address)
        except pygatt.exceptions.NotConnectedError:
            e_msg = "Unable to connect to device at address {}" \
                    .format(self.address)
            raise(IOError(e_msg))

        for uuid in self.device_params["ch_uuids"]:
            self.device.subscribe(uuid, callback=self._transmit_packet)

    def start(self):
        """

        """
        self.sample_index = 0
        self.last_tm = 0
        self.start_time = self.time_func()

        ble_params = self.device_params["ble"]
        self.device.char_write_handle(ble_params["handle"],
                                      value=ble_params["stream_on"],
                                      wait_for_response=False)

    def stop(self):
        """

        """
        ble_params = self.device_params["ble"]
        self.device.char_write_handle(handle=ble_params["handle"],
                                      value=ble_params["stream_off"],
                                      wait_for_response=False)

    def disconnect(self):
        """

        """
        self.device.disconnect()
        self.adapter.stop()

    def _add_device_info(self):
        self.info.desc().append_child_value("manufacturer",
                                            self.device_params["manufacturer"])

        self.channels = self.info.desc().append_child("channels")
        for ch_name in self.device_params["ch_names"]:
            self.channels.append_child("channel") \
                .append_child_value("label", ch_name) \
                .append_child_value("unit", self.device_params["units"]) \
                .append_child_value("type", self.stream_params["type"])

    def _get_device_address(self, name):
        list_devices = self.adapter.scan(timeout=10.5)
        for device in list_devices:
            if name in device['name']:
                return device['address']
        raise(ValueError("No devices found with name `{}`".format(name)))

    def _set_packet_format(self):
        dtypes = self.device_params["packet_dtypes"]
        self.packet_format = dtypes["index"] + \
                             (',' + dtypes["ch_value"]) * self.chunk_size

    def _transmit_packet(self, handle, data):
        """TODO: Move bit locations to Muse parameters."""
        timestamp = self.time_func()
        index = int((handle - 32) / 3)

        tm, d = self._unpack_channel(data)

        if self.last_tm == 0:
            self.last_tm = tm - 1

        self.data[index] = d
        self.timestamps[index] = timestamp

        # if last channel in chunk
        if handle == 35:
            if tm != self.last_tm + 1:
                print("Missing sample {} : {}".format(tm, self.last_tm))
            self.last_tm = tm

            # sample indices
            sample_indices = np.arange(self.chunk_size) + self.sample_index
            self.sample_index += self.chunk_size

            timestamps = sample_indices / self.info.nominal_srate() \
                         + self.start_time

            self._push_chunk(self.data, timestamps)
            self._init_sample()

    def _unpack_channel(self, packet):
        packet_bits = bitstring.Bits(bytes=packet)
        unpacked = packet_bits.unpack(self.packet_format)

        packet_index = unpacked[0]
        packet_values = np.array(unpacked[1:])
        packet_values = 0.48828125 * (packet_values - 2048)

        return packet_index, packet_values


class DummyStreamer(OutletStreamer):
    """

    """

    def __init__(self, dummy_func=None, dur=60, sfreq=256, n_chan=4,
                 csv_file=None, autostart=True, **kwargs):
        """

        """
        OutletStreamer.__init__(self, **kwargs)

        self._thread = threading.Thread(target=self._stream)

        # generate or load fake data
        if csv_file is None:
            self.sfreq = sfreq
            self.n_chan = n_chan
            self.fake_data = self.gen_fake_data(dur)
        else:
            self.csv_file = csv_file
            # TODO:load csv file to np array

        self._proceed = True
        if autostart:
            self._thread.start()

    @classmethod
    def impersonate_device(cls, device, dummy_func=None, dur=60, csv_file=None,
                           **kwargs):
        """

        Args:

        """
        return cls(dur=dur,
                   sfreq=device.STREAM_PARAMS['nominal_srate'],
                   n_chan=device.STREAM_PARAMS['channel_count'],
                   device_params=device.PARAMS,
                   stream_params=device.STREAM_PARAMS,
                   **kwargs)

    def _stream(self):
        while self._proceed:
            self.start_time = self.time_func()
            chunk_inds = np.arange(0, len(self.fake_data.T), self.chunk_size)
            for chunk_ind in chunk_inds:
                self.make_chunk(chunk_ind)
                self._push_chunk(self.data, self.timestamps)
                # force sampling rate
                sec_per_chunk = 1/(self.srate/self.chunk_size)
                time.sleep(sec_per_chunk)

    def gen_fake_data(self, dur, freqs=[5, 10, 12, 20]):
        n_fake_samples = dur*self.srate
        x = np.arange(0, n_fake_samples)
        a_freqs = 2*np.pi*np.array(freqs)
        y = np.zeros((self.n_chan, len(x)))

        # sum frequencies with random amplitudes
        for freq in a_freqs:
            y += np.random.randint(1,5)*np.sin(freq*x)

        noise = np.random.normal(0, 1, (self.n_chan, n_fake_samples))
        fake_data = y + noise

        return fake_data

    def make_chunk(self, chunk_ind):
        self.data = self.fake_data[:,chunk_ind:chunk_ind+self.chunk_size]
        # TODO: more realistic timestamps
        timestamp = self.time_func()
        self.timestamps = np.array([timestamp]*self.chunk_size)
