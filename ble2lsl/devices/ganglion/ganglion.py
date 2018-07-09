"""Interfacing parameters for the OpenBCI Ganglion Board."""

from ble2lsl.devices.device import BasePacketHandler
from ble2lsl.utils import bad_data_size

import struct

import numpy as np
from pygatt import BLEAddressType

PARAMS = dict(
    manufacturer='OpenBCI',
    units='microvolts',
    ch_names=('A', 'B', 'C', 'D'),  # placement-dependent...
    chunk_size=1,
    ble=dict(
        address_type=BLEAddressType.random,
        # service='fe84',
        interval_min=6,  # OpenBCI suggest 9
        interval_max=11,  # suggest 10
        receive=['2d30c082f39f4ce6923f3484ea480596'],
        send="2d30c083f39f4ce6923f3484ea480596",
        stream_on=b'b',
        stream_off=b's',
        # accelerometer_on=b'n',
        # accelerometer_off=b'N',
        # impedance_on=b'z',
        # impedance_off=b'Z',
        # disconnect="2d30c084f39f4ce6923f3484ea480596",
    ),
)
"""General OpenBCI Ganglion parameters, including BLE characteristics."""

LSL_INFO = dict(
    name='Ganglion',
    type='EEG',
    channel_count=4,
    nominal_srate=200,
    channel_format='float32'
)
"""OpenBCI Ganglion parameters for constructing `pylsl.StreamInfo`."""

INT_SIGN_BYTE = (b'\x00', b'\xff')
SCALE_FACTOR_EEG = 1200 / (8388607.0 * 1.5 * 51.0)  # microvolts per count
SCALE_FACTOR_ACCEL = 0.000016  # G per count


class PacketHandler(BasePacketHandler):
    """Process packets from the OpenBCI Ganglion into chunks."""

    def __init__(self, output_queue, **kwargs):
        super().__init__(device_params=PARAMS,
                         output_queue=output_queue,
                         **kwargs)
        # holds samples until OpenBCIBoard claims them
        # detect gaps between packets
        self._last_id = -1
        # 18bit data got here and then accelerometer with it
        self.last_accelerometer = [0, 0, 0]
        # when the board is manually set in the right mode, impedance will be measured. 4 channels + ref
        self.last_impedance = [0, 0, 0, 0, 0]

        # byte ID ranges for parsing function selection
        self._byte_id_ranges = {(101, 200): self.parse_compressed_19bit,
                                (0, 0): self.parse_uncompressed,
                                (206, 207): self.print_ascii,
                                (1, 100): self.parse_compressed_18bit,
                                (201, 205): self.parse_impedance,
                                (208, -1): self.unknown_packet_warning}

    def process_packet(self, packet, handle):
        """Process incoming data packet.

        Calls the corresponding parsing function depending on packet format.
        """
        start_byte = packet[0]
        for r in self._byte_id_ranges:
            if start_byte >= r[0] and start_byte <= r[1]:
                self._byte_id_ranges[r](start_byte, packet[1:])
                break

    def push_sample(self, sample_id, chan_data, aux_data, imp_data):
        """Add a sample to inner stack, setting ID and dealing with scaling if necessary. """
        if self.scaling_output:
            chan_data *= SCALE_FACTOR_EEG
            # aux_data = np.array(aux_data) * SCALE_FACTOR_ACCEL_G_per_count
        self.update_packets_count(sample_id)
        self._sample_idxs[0] = self._count_id
        self._data[:] = chan_data
        self._output_queue.put(self.output)

    def update_packets_count(self, sample_id):
        """Update last packet ID and dropped packets"""
        if self._last_id == -1:
            self._last_id = sample_id
            self._count_id = 1
            return
        # ID loops every 101 packets (201 samples)
        if sample_id > self._last_id:
            self._count_id += sample_id - self._last_id
        else:
            self._count_id += sample_id - self._last_id + 201
        self._last_id = sample_id

    def unknown_packet_warning(self, start_byte, packet):
        """Print if incoming byte ID is unknown."""
        print("Warning: unknown type of packet: {}".format(start_byte))

    def print_ascii(self, start_byte, packet):
        """Print verbose ASCII data.

        TODO:
            * optional log file
        """
        print("%\t" + str(packet))
        if start_byte == 207:
            print("$$$\n")

    def parse_uncompressed(self, packet_id, packet):
        """Parse a raw uncompressed packet."""
        if bad_data_size(packet, 19, "uncompressed data"):
            return
        # 4 channels of 24bits, take values one by one
        chan_data = [int_from_24bits(packet[i:i+3]) for i in range(0, 12, 3)]
        chan_data = np.array([chan_data], dtype=np.float32).T
        # save uncompressed raw channel for future use and append whole sample
        self.push_sample(packet_id, chan_data, self.last_accelerometer,
                         self.last_impedance)

    def update_data_with_deltas(self, packet_id, deltas):
        for delta_id in [0, 1]:
            # convert from packet to sample ID
            sample_id = (packet_id - 1) * 2 + delta_id + 1
            # 19bit packets hold deltas between two samples
            # TODO: use more broadly numpy
            chan_data = self._data - deltas[delta_id, :].reshape(4, 1)
            self.push_sample(sample_id, chan_data, self.last_accelerometer,
                             self.last_impedance)

    def parse_compressed_19bit(self, packet_id, packet):
        """Parse a 19-bit compressed packet without accelerometer data."""
        if bad_data_size(packet, 19, "19-bit compressed data"):
            return

        packet_id -= 100
        # should get 2 by 4 arrays of uncompressed data
        deltas = decompress_deltas_19bit(packet)
        self.update_data_with_deltas(packet_id, deltas)

    def parse_compressed_18bit(self, packet_id, packet):
        """ Dealing with "18-bit compression without Accelerometer" """
        if bad_data_size(packet, 19, "18-bit compressed data"):
            return

        # set appropriate accelerometer byte
        self.last_accelerometer[packet_id % 10 - 1] = int8_from_byte(packet[18])

        # deltas: should get 2 by 4 arrays of uncompressed data
        deltas = decompress_deltas_18bit(packet[:-1])
        self.update_data_with_deltas(packet_id, deltas)

    def parse_impedance(self, packet_id, packet):
        """Parse impedance data.

        After turning on impedance checking, takes a few seconds to complete.
        """
        if packet[-2:] != 'Z\n':
            print("Wrong format for impedance: not ASCII ending with 'Z\\n'")

        # convert from ASCII to actual value
        imp_value = int(packet[:-2])
        # from 201 to 205 codes to the right array size
        self.last_impedance[packet_id - 201] = imp_value
        self.push_sample(packet_id - 200, self._data,
                         self.last_accelerometer, self.last_impedance)

def int_from_24bits(unpacked):
    """Convert 24-bit data coded on 3 bytes to a proper integer."""
    if bad_data_size(unpacked, 3, "3-byte buffer"):
        raise ValueError("Bad input size for byte conversion.")

    # FIXME: quick'n dirty, unpack wants strings later on
    int_bytes = INT_SIGN_BYTE[unpacked[0] > 127] + struct.pack('3B', *unpacked)

    # unpack little endian(>) signed integer(i) (-> platform independent)
    int_unpacked = struct.unpack('>i', int_bytes)[0]

    return int_unpacked


def int32_from_19bit(three_byte_buffer):
    """Convert 19-bit data coded on 3 bytes to a proper integer."""
    if bad_data_size(three_byte_buffer, 3, "3-byte buffer"):
        raise ValueError("Bad input size for byte conversion.")

    # if LSB is 1, negative number
    if three_byte_buffer[2] & 0x01 > 0:
        prefix = 0b1111111111111
        int32 = ((prefix << 19) | (three_byte_buffer[0] << 16)
                 | (three_byte_buffer[1] << 8) | three_byte_buffer[2]) \
                | ~0xFFFFFFFF
    else:
        prefix = 0
        int32 = (prefix << 19) | (three_byte_buffer[0] << 16) \
                | (three_byte_buffer[1] << 8) | three_byte_buffer[2]

    return int32


def int32_from_18bit(three_byte_buffer):
    """Convert 18-bit data coded on 3 bytes to a proper integer."""
    if bad_data_size(three_byte_buffer, 3, "3-byte buffer"):
        raise ValueError("Bad input size for byte conversion.")

    # if LSB is 1, negative number, some hasty unsigned to signed conversion to do
    if three_byte_buffer[2] & 0x01 > 0:
        prefix = 0b11111111111111
        int32 = ((prefix << 18) | (three_byte_buffer[0] << 16)
                 | (three_byte_buffer[1] << 8) | three_byte_buffer[2]) \
                | ~0xFFFFFFFF
    else:
        prefix = 0
        int32 = (prefix << 18) | (three_byte_buffer[0] << 16) \
                | (three_byte_buffer[1] << 8) | three_byte_buffer[2]

    return int32


def int8_from_byte(byte):
    """Convert one byte to signed integer."""
    if byte > 127:
        return (256 - byte) * (-1)
    else:
        return byte


def decompress_deltas_19bit(buffer):
    """Called to when a compressed packet is received.
    buffer: Just the data portion of the sample. So 19 bytes.
    return {Array} - An array of deltas of shape 2x4 (2 samples per packet and 4 channels per sample.)
    """
    if bad_data_size(buffer, 19, "19-byte compressed packet"):
        raise ValueError("Bad input size for byte conversion.")

    deltas = np.zeros((2, 4))

    # Sample 1 - Channel 1
    minibuf = [(buffer[0] >> 5),
               ((buffer[0] & 0x1F) << 3 & 0xFF) | (buffer[1] >> 5),
               ((buffer[1] & 0x1F) << 3 & 0xFF) | (buffer[2] >> 5)]
    deltas[0][0] = int32_from_19bit(minibuf)

    # Sample 1 - Channel 2
    minibuf = [(buffer[2] & 0x1F) >> 2,
               (buffer[2] << 6 & 0xFF) | (buffer[3] >> 2),
               (buffer[3] << 6 & 0xFF) | (buffer[4] >> 2)]
    deltas[0][1] = int32_from_19bit(minibuf)

    # Sample 1 - Channel 3
    minibuf = [((buffer[4] & 0x03) << 1 & 0xFF) | (buffer[5] >> 7),
               ((buffer[5] & 0x7F) << 1 & 0xFF) | (buffer[6] >> 7),
               ((buffer[6] & 0x7F) << 1 & 0xFF) | (buffer[7] >> 7)]
    deltas[0][2] = int32_from_19bit(minibuf)

    # Sample 1 - Channel 4
    minibuf = [((buffer[7] & 0x7F) >> 4),
               ((buffer[7] & 0x0F) << 4 & 0xFF) | (buffer[8] >> 4),
               ((buffer[8] & 0x0F) << 4 & 0xFF) | (buffer[9] >> 4)]
    deltas[0][3] = int32_from_19bit(minibuf)

    # Sample 2 - Channel 1
    minibuf = [((buffer[9] & 0x0F) >> 1),
               (buffer[9] << 7 & 0xFF) | (buffer[10] >> 1),
               (buffer[10] << 7 & 0xFF) | (buffer[11] >> 1)]
    deltas[1][0] = int32_from_19bit(minibuf)

    # Sample 2 - Channel 2
    minibuf = [((buffer[11] & 0x01) << 2 & 0xFF) | (buffer[12] >> 6),
               (buffer[12] << 2 & 0xFF) | (buffer[13] >> 6),
               (buffer[13] << 2 & 0xFF) | (buffer[14] >> 6)]
    deltas[1][1] = int32_from_19bit(minibuf)

    # Sample 2 - Channel 3
    minibuf = [((buffer[14] & 0x38) >> 3),
               ((buffer[14] & 0x07) << 5 & 0xFF) | ((buffer[15] & 0xF8) >> 3),
               ((buffer[15] & 0x07) << 5 & 0xFF) | ((buffer[16] & 0xF8) >> 3)]
    deltas[1][2] = int32_from_19bit(minibuf)

    # Sample 2 - Channel 4
    minibuf = [(buffer[16] & 0x07), buffer[17], buffer[18]]
    deltas[1][3] = int32_from_19bit(minibuf)

    return deltas


def decompress_deltas_18bit(buffer):
    """Called to when a compressed packet is received.
    buffer: Just the data portion of the sample. So 19 bytes.
    return {Array} - An array of deltas of shape 2x4 (2 samples per packet and 4 channels per sample.)
    """
    if bad_data_size(buffer, 18, "18-byte compressed packet"):
        raise ValueError("Bad input size for byte conversion.")

    deltas = np.zeros((2, 4))

    # Sample 1 - Channel 1
    minibuf = [(buffer[0] >> 6),
               ((buffer[0] & 0x3F) << 2 & 0xFF) | (buffer[1] >> 6),
               ((buffer[1] & 0x3F) << 2 & 0xFF) | (buffer[2] >> 6)]
    deltas[0][0] = int32_from_18bit(minibuf)

    # Sample 1 - Channel 2
    minibuf = [(buffer[2] & 0x3F) >> 4,
               (buffer[2] << 4 & 0xFF) | (buffer[3] >> 4),
               (buffer[3] << 4 & 0xFF) | (buffer[4] >> 4)]
    deltas[0][1] = int32_from_18bit(minibuf)

    # Sample 1 - Channel 3
    minibuf = [(buffer[4] & 0x0F) >> 2,
               (buffer[4] << 6 & 0xFF) | (buffer[5] >> 2),
               (buffer[5] << 6 & 0xFF) | (buffer[6] >> 2)]
    deltas[0][2] = int32_from_18bit(minibuf)

    # Sample 1 - Channel 4
    minibuf = [(buffer[6] & 0x03), buffer[7], buffer[8]]
    deltas[0][3] = int32_from_18bit(minibuf)

    # Sample 2 - Channel 1
    minibuf = [(buffer[9] >> 6),
               ((buffer[9] & 0x3F) << 2 & 0xFF) | (buffer[10] >> 6),
               ((buffer[10] & 0x3F) << 2 & 0xFF) | (buffer[11] >> 6)]
    deltas[1][0] = int32_from_18bit(minibuf)

    # Sample 2 - Channel 2
    minibuf = [(buffer[11] & 0x3F) >> 4,
               (buffer[11] << 4 & 0xFF) | (buffer[12] >> 4),
               (buffer[12] << 4 & 0xFF) | (buffer[13] >> 4)]
    deltas[1][1] = int32_from_18bit(minibuf)

    # Sample 2 - Channel 3
    minibuf = [(buffer[13] & 0x0F) >> 2,
               (buffer[13] << 6 & 0xFF) | (buffer[14] >> 2),
               (buffer[14] << 6 & 0xFF) | (buffer[15] >> 2)]
    deltas[1][2] = int32_from_18bit(minibuf)

    # Sample 2 - Channel 4
    minibuf = [(buffer[15] & 0x03), buffer[16], buffer[17]]
    deltas[1][3] = int32_from_18bit(minibuf)

    return deltas
