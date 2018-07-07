"""Interfacing parameters for the Muse headband (2016 version)."""

from ble2lsl.devices.device import BasePacketHandler

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
        interval_min=60,  # pygatt default, seems fine
        interval_max=76,  # pygatt default
        # characteristic UUIDs
        receive=['273e0003-4c4d-454d-96be-f03bac821358',
                 '273e0004-4c4d-454d-96be-f03bac821358',
                 '273e0005-4c4d-454d-96be-f03bac821358',
                 '273e0006-4c4d-454d-96be-f03bac821358',
                 '273e0007-4c4d-454d-96be-f03bac821358'],
        send='273e0001-4c4d-454d-96be-f03bac821358',
        # gyro='273e0009-4c4d-454d-96be-f03bac821358',
        # accelerometer='273e000a-4c4d-454d-96be-f03bac821358',
        # telemetry='273e000b-4c4d-454d-96be-f03bac821358',
        # commands (write to send characteristic)
        stream_on=(0x02, 0x64, 0x0a),  # b'd'
        stream_off=(0x02, 0x68, 0x0a),  # ?
        # keep_alive=(0x02, 0x6b, 0x0a), # (?) b'k'
        # request_info=(0x03, 0x76, 0x31, 0x0a),
        # request_status=(0x02, 0x73, 0x0a),
        # reset=(0x03, 0x2a, 0x31, 0x0a)
    ),
)
"""General Muse headset parameters, including BLE characteristics."""

LSL_INFO = dict(
    name='Muse',
    type='EEG',
    channel_count=5,
    nominal_srate=256,
    channel_format='float32',
)
"""Muse headset parameters for constructing `pylsl.StreamInfo`."""

PACKET_FORMAT = 'uint:16' + ',uint:12' * PARAMS["chunk_size"]
HANDLE_CH_IDXS = {32: 0, 35: 1, 38: 2, 41: 3, 44: 4}
HANDLE_RECEIVE_ORDER = [44, 41, 38, 32, 35]


def convert_count_to_uvolts(value):
    return 0.48828125 * (value - 2048)


class PacketHandler(BasePacketHandler):
    """Process packets from the Muse 2016 headset into chunks.
    """

    def __init__(self, output_queue, **kwargs):
        super().__init__(device_params=PARAMS,
                         output_queue=output_queue,
                         **kwargs)

    def process_packet(self, data, handle):
        # TODO: last handle then send (flag?)
        packet_idx, ch_values = self._unpack_channel(data, PACKET_FORMAT)
        idx = HANDLE_CH_IDXS[handle]
        self._data[idx] = ch_values
        self._sample_idxs[idx] = packet_idx
        if handle == HANDLE_RECEIVE_ORDER[-1]:
            self._output_queue.put(self.output)

    def _unpack_channel(self, packet, PACKET_FORMAT):
        """Parse the bitstrings received over BLE."""
        packet_bits = bitstring.Bits(bytes=packet)
        unpacked = packet_bits.unpack(PACKET_FORMAT)

        packet_index = unpacked[0]
        packet_values = np.array(unpacked[1:])
        if self.scaling_output:
            packet_values = convert_count_to_uvolts(packet_values)

        return packet_index, packet_values
