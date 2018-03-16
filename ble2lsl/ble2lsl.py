"""Interface between BLE/GATT and the Lab Streaming Layer.

TODO:
    * Subclass LSLOutletStreamer instead of using Muse as defaults.
"""

from ble2lsl.devices.muse import MUSE_PARAMS, MUSE_STREAM_PARAMS

import time

import bitstring
import numpy as np
import pygatt
import pylsl as lsl
import threading


class LSLOutletStreamer():
    """

    """

    def __init__(self, device_params=None, stream_params=None, interface=None,
                 address=None, backend='bgapi', autostart=True, chunk_size=12,
                 time_func=time.time):
        """

        """
        if device_params is None:
            device_params = MUSE_PARAMS
        if stream_params is None:
            stream_params = MUSE_STREAM_PARAMS

        self.device_params = device_params
        self.stream_params = stream_params
        self.chunk_size = chunk_size
        self.interface = interface
        self.address = address
        self.time_func = time_func

        # initialize gatt adapter
        if backend == 'gatt':
            self.interface = self.interface or 'hci0'
            self.adapter = pygatt.GATTToolBackend(self.interface)
        elif backend == 'bgapi':
            self.adapter = pygatt.BGAPIBackend(serial_port=self.interface)
        else:
            raise(ValueError("Invalid backend specified; use bgapi or gatt."))
        self.backend = backend

        # construct LSL StreamInfo and StreamOutlet
        self._init_stream_info(stream_params)
        self.outlet = lsl.StreamOutlet(self.info, chunk_size=chunk_size,
                                       max_buffered=360)
        self._set_packet_format()

        if autostart:
            self.connect()
            self.start()

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

        self._init_sample()

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

    def _get_device_address(self, name):
        list_devices = self.adapter.scan(timeout=10.5)
        for device in list_devices:
            if name in device['name']:
                return device['address']
        raise(ValueError("No devices found with name `{}`".format(name)))

    def _init_stream_info(self, stream_params):
        self.info = lsl.StreamInfo(**stream_params, source_id="MuseNone")

        self.info.desc().append_child_value("manufacturer",
                                            self.device_params["manufacturer"])

        self.channels = self.info.desc().append_child("channels")
        for ch_name in self.device_params["ch_names"]:
            self.channels.append_child("channel") \
                .append_child_value("label", ch_name) \
                .append_child_value("unit", self.device_params["units"]) \
                .append_child_value("type", stream_params["type"])

    def _set_packet_format(self):
        dtypes = self.device_params["packet_dtypes"]
        n_chan = self.info.channel_count()
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

    def _init_sample(self):
        self.timestamps = np.zeros(self.info.channel_count())
        self.data = np.zeros((self.info.channel_count(), self.chunk_size))

    def _push_chunk(self, channels, timestamps):
        for sample in range(self.chunk_size):
            self.outlet.push_sample(channels[:, sample], timestamps[sample])


class LSLOutletDummy(threading.Thread):
    def __init__(self, csv_file=None, dur=60, device_params=None,
                 stream_params=None, autostart=True, chunk_size=12,
                 time_func=time.time):
        threading.Thread.__init__(self)
        if device_params is None:
            device_params = MUSE_PARAMS
        if stream_params is None:
            stream_params = MUSE_STREAM_PARAMS

        self.device_params = device_params
        self.stream_params = stream_params
        self.chunk_size = chunk_size
        self.time_func = time_func
        self.csv_file = csv_file

        self.srate = self.stream_params['nominal_srate']
        self.n_chan = self.stream_params['channel_count']

        # construct LSL StreamInfo and StreamOutlet
        self._init_stream_info(stream_params)
        self.outlet = lsl.StreamOutlet(self.info, chunk_size=chunk_size,
                                       max_buffered=360)

        # generate or load fake data
        if csv_file is None:
            self.fake_data = self.gen_fake_data(dur)

        else:
            # TODO:load csv file to np array
            # get params from somewhere? MUSE_STREAM_PARAMS for now
            pass

        if autostart:
            self.start()

    def run(self):
        self.start_time = self.time_func()
        chunk_inds = np.arange(0, len(self.fake_data.T), self.chunk_size)
        for chunk_ind in chunk_inds:
            self.make_chunk(chunk_ind)
            self._push_chunk(self.data, self.timestamps)
            # force sampling rate
            sec_per_chunk = 1/(self.srate/self.chunk_size)
            time.sleep(sec_per_chunk)
        # hacky way to run indefinitely
        self.rerun()

    def rerun(self):
        self.run()

    def _init_stream_info(self, stream_params):
        self.info = lsl.StreamInfo(**stream_params, source_id="MuseNone")

        self.info.desc().append_child_value("manufacturer",
                                            self.device_params["manufacturer"])

        self.channels = self.info.desc().append_child("channels")
        for ch_name in self.device_params["ch_names"]:
            self.channels.append_child("channel") \
                .append_child_value("label", ch_name) \
                .append_child_value("unit", self.device_params["units"]) \
                .append_child_value("type", stream_params["type"])

    def _init_sample(self):
        self.timestamps = np.zeros(self.info.channel_count())
        self.data = np.zeros((self.info.channel_count(), self.chunk_size))

    def _push_chunk(self, channels, timestamps):
        for sample in range(self.chunk_size):
            self.outlet.push_sample(channels[:, sample], timestamps[sample])

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
