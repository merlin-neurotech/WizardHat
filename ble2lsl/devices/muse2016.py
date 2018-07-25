"""Interfacing parameters for the Muse headband (2016 version).

More information on the data provided by the Muse 2016 headband can be found
at `Available Data - Muse Direct`_

TODO:
    * Figure out maximum string size for status messages, or split into fields
      (can't send dict over LSL)
    * return standard acceleration units and not g's...
    * verify telemetry and IMU conversions and units
    * DRL/REF characteristic
    * don't use lambdas for CONVERT_FUNCS?
    * save Muse address to minimize connect time?
    * packet ID rollover (uint16) -- generalize in device file?

.. _Available Data - Muse Direct:
   http://developer.choosemuse.com/tools/windows-tools/available-data-muse-direct
"""

from ble2lsl.devices.device import BasePacketHandler
from ble2lsl.utils import dict_partial_from_keys

import bitstring
import numpy as np
from pygatt import BLEAddressType

NAME = 'Muse'
MANUFACTURER = 'Interaxon'

STREAMS = ['EEG', 'accelerometer', 'gyroscope', 'telemetry', 'status']
"""Data sources provided by the Muse 2016 headset."""

DEFAULT_SUBSCRIPTIONS = STREAMS
"""Sources to which to subscribe by default."""

# for constructing dicts with STREAMS as keys
streams_dict = dict_partial_from_keys(STREAMS)

PARAMS = dict(
    streams=dict(
        type=streams_dict(STREAMS),  # identity mapping. best solution?
        channel_count=streams_dict([5, 3, 3, 4, 1]),
        nominal_srate=streams_dict([256, 52, 52, 0.1, 0.0]),
        channel_format=streams_dict(['float32', 'float32', 'float32',
                                     'float32', 'string']),
        numpy_dtype=streams_dict(['float32', 'float32', 'float32', 'float32',
                                  'object']),
        units=streams_dict([('uV',) * 5,
                            ('g\'s',) * 3,
                            ('deg/s',) * 3,
                            ('%', 'mV', 'mV', 'C'),
                            ('',)]),
        ch_names=streams_dict([('TP9', 'AF7', 'AF8', 'TP10', 'Right AUX'),
                               ('x', 'y', 'z'),
                               ('x', 'y', 'z'),
                               ('battery', 'fuel_gauge', 'adc_volt',
                                'temperature'),
                               ('message',)]),
        chunk_size=streams_dict([12, 3, 3, 1, 1]),
    ),

    ble=dict(
        address_type=BLEAddressType.public,
        interval_min=60,  # pygatt default, seems fine
        interval_max=76,  # pygatt default

        # receive characteristic UUIDs
        EEG=['273e0003-4c4d-454d-96be-f03bac821358',
             '273e0004-4c4d-454d-96be-f03bac821358',
             '273e0005-4c4d-454d-96be-f03bac821358',
             '273e0006-4c4d-454d-96be-f03bac821358',
             '273e0007-4c4d-454d-96be-f03bac821358'],
        # reference=['273e0008-4c4d-454d-96be-f03bac821358'],
        accelerometer='273e000a-4c4d-454d-96be-f03bac821358',
        gyroscope='273e0009-4c4d-454d-96be-f03bac821358',
        telemetry='273e000b-4c4d-454d-96be-f03bac821358',
        status='273e0001-4c4d-454d-96be-f03bac821358',  # same as send

        # send characteristic UUID and commands
        send='273e0001-4c4d-454d-96be-f03bac821358',
        stream_on=(0x02, 0x64, 0x0a),  # b'd'
        stream_off=(0x02, 0x68, 0x0a),  # ?
        # keep_alive=(0x02, 0x6b, 0x0a), # (?) b'k'
        # request_info=(0x03, 0x76, 0x31, 0x0a),
        # request_status=(0x02, 0x73, 0x0a),
        # reset=(0x03, 0x2a, 0x31, 0x0a)
    )
)
"""Muse 2016 LSL- and BLE-related parameters."""

HANDLE_NAMES = {14: "status", 26: "telemetry", 23: "accelerometer",
                20: "gyroscope", 32: "EEG", 35: "EEG", 38: "EEG", 41: "EEG",
                44: "EEG"}
"""Stream name associated with each packet handle."""

PACKET_FORMATS = streams_dict(['uint:16' + ',uint:12' * 12,
                               'uint:16' + ',int:16' * 9,
                               'uint:16' + ',int:16' * 9,
                               'uint:16' + ',uint:16' * 4,
                               ','.join(['uint:8'] * 20)])
"""Byte formats of the incoming packets."""

CONVERT_FUNCS = streams_dict([lambda data: 0.48828125 * (data - 2048),
                              lambda data: 0.0000610352 * data.reshape((3, 3)),
                              lambda data: 0.0074768 * data.reshape((3, 3)),
                              lambda data: np.array([data[0] / 512,
                                                     2.2 * data[1],
                                                     data[2], data[3]]).reshape((1, 4)),
                              lambda data: None])
"""Functions to render unpacked data into the appropriate shape and units."""

EEG_HANDLE_CH_IDXS = {32: 0, 35: 1, 38: 2, 41: 3, 44: 4}
EEG_HANDLE_RECEIVE_ORDER = [44, 41, 38, 32, 35]
"""Channel indices and receipt order of EEG packets."""


class PacketHandler(BasePacketHandler):
    """Process packets from the Muse 2016 headset into chunks."""

    def __init__(self, streamer, **kwargs):
        super().__init__(PARAMS["streams"], streamer, **kwargs)

        if "status" in self._streamer.subscriptions:
            self._chunks["status"][0] = ""
            self._chunk_idxs["status"] = -1

    def process_packet(self, handle, packet):
        """Unpack, convert, and return packet contents."""
        name = HANDLE_NAMES[handle]
        unpacked = _unpack(packet, PACKET_FORMATS[name])

        if name not in self._streamer.subscriptions:
            return

        if name == "status":
            self._process_status(unpacked)
        else:
            data = np.array(unpacked[1:],
                            dtype=PARAMS["streams"]["numpy_dtype"][name])

            if name == "EEG":
                idx = EEG_HANDLE_CH_IDXS[handle]
                self._chunks[name][:, idx] = CONVERT_FUNCS[name](data)
                if not handle == EEG_HANDLE_RECEIVE_ORDER[-1]:
                    return
            else:
                try:
                    self._chunks[name][:, :] = CONVERT_FUNCS[name](data)
                except ValueError:
                    print(name)

            self._chunk_idxs[name] = unpacked[0]
            self._enqueue_chunk(name)

    def _process_status(self, unpacked):
        message_chars = [chr(i) for i in unpacked[1:]]
        status_message_partial = "".join(message_chars)[:unpacked[0]]
        self._chunks["status"] += status_message_partial.replace('\n', '')
        if status_message_partial[-1] == '}':
            # ast.literal_eval(self._message))
            # parse and enqueue dict
            self._enqueue_chunk("status")
            self._chunks["status"][0] = ""


def _unpack(packet, packet_format):
    packet_bits = bitstring.Bits(bytes=packet)
    unpacked = packet_bits.unpack(packet_format)
    return unpacked
