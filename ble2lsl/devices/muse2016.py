"""Interfacing parameters for the Muse headband (2016 version)."""

from pygatt import BLEAddressType
import bitstring
PARAMS = dict(
    manufacturer='Muse',
    units='microvolts',
    ch_names=('TP9', 'AF7', 'AF8', 'TP10', 'Right AUX'),
    chunk_size=12,
    packet_dtypes=dict(index='uint:16', ch_value='uint:12'),
    ble=dict(
        uuid=[
            '273e0003-4c4d-454d-96be-f03bac821358',
            '273e0004-4c4d-454d-96be-f03bac821358',
            '273e0005-4c4d-454d-96be-f03bac821358',
            '273e0006-4c4d-454d-96be-f03bac821358',
            '273e0007-4c4d-454d-96be-f03bac821358',
        ],
        address_type=BLEAddressType.public,
        send='273e0001-4c4d-454d-96be-f03bac821358',
        stream_on=(0x02, 0x64, 0x0a),
        stream_off=(0x02, 0x68, 0x0a),
    ),
)
"""General Muse headset parameters."""


STREAM_PARAMS = dict(
    name='Muse',
    type='EEG',
    channel_count=5,
    nominal_srate=256,
    channel_format='float32',
)
"""Muse headset parameters for constructing `pylsl.StreamInfo`."""
import numpy as np

class PacketManager():
    def __init__(self, scaling_output = True):
        self.scaling_output = scaling_output

    def parse(self,data,packet_format):
 

        tm, d = self._unpack_channel(data,packet_format)
        self.sample = [tm, d]

    def _unpack_channel(self, packet, packet_format):
        """Parse the bitstrings received over BLE."""
        packet_bits = bitstring.Bits(bytes=packet)
        unpacked = packet_bits.unpack(packet_format)

        packet_index = unpacked[0]
        packet_values = np.array(unpacked[1:])
        if self.scaling_output:
            packet_values = 0.48828125 * (packet_values - 2048)

        return packet_index, packet_values


