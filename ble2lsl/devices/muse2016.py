"""Interfacing parameters for the Muse headband (2016 version)."""

from ble2lsl.utils import invert_map

import bitstring
import numpy as np
from pygatt import BLEAddressType

PARAMS = dict(
    manufacturer='Muse',
    units='microvolts',
    ch_names=('TP9', 'AF7', 'AF8', 'TP10', 'Right AUX'),
    chunk_size=12,
    ble=dict(
        address_type=BLEAddressType.public,
        interval_min=60,  # pygatt default, needs testing?
        interval_max=76,  # pygatt default
        receive=['273e0003-4c4d-454d-96be-f03bac821358',
                 '273e0004-4c4d-454d-96be-f03bac821358',
                 '273e0005-4c4d-454d-96be-f03bac821358',
                 '273e0006-4c4d-454d-96be-f03bac821358',
                 '273e0007-4c4d-454d-96be-f03bac821358'],
        send='273e0001-4c4d-454d-96be-f03bac821358',
        stream_on=(0x02, 0x64, 0x0a),
        stream_off=(0x02, 0x68, 0x0a),
    ),
)
"""General Muse headset parameters, including BLE characteristics."""

STREAM_PARAMS = dict(
    name='Muse',
    type='EEG',
    channel_count=5,
    nominal_srate=256,
    channel_format='float32',
)
"""Muse headset parameters for constructing `pylsl.StreamInfo`."""

PACKET_FORMAT = 'uint:16' + (',' + 'uint:12') * PARAMS["chunk_size"]
LAST_HANDLE = 35
PACKET_HANDLES = {32: 0, 35: 1, 38: 2, 41: 3, 44: 4}


def convert_count_to_uvolts(value):
    return 0.48828125 * (value - 2048)


class PacketManager():
    """"""

    def __init__(self, output_queue, scaling_output=True):
        self.scaling_output = scaling_output
        self._data = np.zeros((STREAM_PARAMS["channel_count"],
                               PARAMS["chunk_size"]))
        self._sample_idxs = np.zeros(STREAM_PARAMS["channel_count"])
        self._output_queue = output_queue

    def process_packet(self, data, handle):
        # TODO: last handle then send (flag?)
        packet_idx, ch_values = self._unpack_channel(data, PACKET_FORMAT)
        idx = PACKET_HANDLES[handle]
        self._data[idx] = ch_values
        self._sample_idxs[idx] = packet_idx
        if handle == 35:
            self._output_queue.put(self.output)

    @property
    def output(self):
        return (np.copy(self._sample_idxs), np.copy(self._data))

    def _unpack_channel(self, packet, PACKET_FORMAT):
        """Parse the bitstrings received over BLE."""
        packet_bits = bitstring.Bits(bytes=packet)
        unpacked = packet_bits.unpack(PACKET_FORMAT)

        packet_index = unpacked[0]
        packet_values = np.array(unpacked[1:])
        if self.scaling_output:
            packet_values = convert_count_to_uvolts(packet_values)

        return packet_index, packet_values
