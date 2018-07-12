"""Interfacing parameters for the OpenBCI Ganglion Board."""

from ble2lsl.devices.device import BasePacketHandler
from ble2lsl.utils import bad_data_size, dict_partial_from_keys

import struct
from warnings import warn

import numpy as np
from pygatt import BLEAddressType

NAME = "Ganglion"

MANUFACTURER = "OpenBCI"

STREAMS = ["EEG", "accelerometer", "messages"]
"""Data provided by the OpenBCI Ganglion, and available for subscription."""

DEFAULT_SUBSCRIPTIONS = ["EEG", "messages"]
"""Streams to which to subscribe by default."""

# for constructing dicts with STREAMS as keys
streams_dict = dict_partial_from_keys(STREAMS)

PARAMS = dict(
    streams=dict(
        type=streams_dict(STREAMS),  # same as stream names
        channel_count=streams_dict([4, 3, 1]),
        nominal_srate=streams_dict([200, 10, 0.0]),
        channel_format=streams_dict(['float32', 'float32', 'string']),
        numpy_dtype=streams_dict(['float32', 'float32', 'object']),
        units=streams_dict([('uV',) * 4, ('g\'s',) * 3, ('',)]),
        ch_names=streams_dict([('A', 'B', 'C', 'D'), ('x', 'y', 'z'),
                               ('message',)]),
        chunk_size=streams_dict([1, 1, 1]),
    ),
    ble=dict(
        address_type=BLEAddressType.random,
        # service='fe84',
        interval_min=6,  # OpenBCI suggest 9
        interval_max=11,  # suggest 10

        # receive characteristic UUIDs
        EEG=["2d30c082f39f4ce6923f3484ea480596"],
        accelerometer='',  # placeholder; already subscribed through eeg
        messages='',  # placeholder; subscription not required

        # send characteristic UUID and commands
        send="2d30c083f39f4ce6923f3484ea480596",
        stream_on=b'b',
        stream_off=b's',
        accelerometer_on=b'n',
        accelerometer_off=b'N',
        # impedance_on=b'z',
        # impedance_off=b'Z',

        # other characteristics
        # disconnect="2d30c084f39f4ce6923f3484ea480596",
    ),
)
"""OpenBCI Ganglion LSL- and BLE-related parameters."""

INT_SIGN_BYTE = (b'\x00', b'\xff')
SCALE_FACTOR = streams_dict([1.2 / (8388608.0 * 1.5 * 51.0),
                             0.016,
                             1  # not used (messages)
                             ])
"""Scale factors for conversion of EEG and accelerometer data to mV."""

ID_TURNOVER = streams_dict([201, 10])
"""The number of samples processed before the packet ID cycles back to zero."""


class PacketHandler(BasePacketHandler):
    """Process packets from the OpenBCI Ganglion into chunks."""

    def __init__(self, streamer, **kwargs):
        super().__init__(PARAMS["streams"], streamer, **kwargs)

        self._sample_ids = streams_dict([-1] * len(STREAMS))

        if "EEG" in self._streamer.subscriptions:
            self._last_eeg_data = np.zeros(self._chunks["EEG"].shape[1])

        if "messages" in self._streamer.subscriptions:
            self._chunks["messages"][0] = ""
            self._chunk_idxs["messages"] = -1

        if "accelerometer" in self._streamer.subscriptions:
            # queue accelerometer_on command
            self._streamer.send_command(PARAMS["ble"]["accelerometer_on"])

        # byte ID ranges for parsing function selection
        self._byte_id_ranges = {(101, 200): self._parse_compressed_19bit,
                                (0, 0): self._parse_uncompressed,
                                (206, 207): self._parse_message,
                                (1, 100): self._parse_compressed_18bit,
                                (201, 205): self._parse_impedance,
                                (208, -1): self._unknown_packet_warning}

    def process_packet(self, handle, packet):
        """Process incoming data packet.

        Calls the corresponding parsing function depending on packet format.
        """
        start_byte = packet[0]
        for r in self._byte_id_ranges:
            if start_byte >= r[0] and start_byte <= r[1]:
                self._byte_id_ranges[r](start_byte, packet[1:])
                break

    def _update_counts_and_enqueue(self, name, sample_id):
        """Update last packet ID and dropped packets"""
        if self._sample_ids[name] == -1:
            self._sample_ids[name] = sample_id
            self._chunk_idxs[name] = 1
            return
        # sample IDs loops every 101 packets
        self._chunk_idxs[name] += sample_id - self._sample_ids[name]
        if sample_id < self._sample_ids[name]:
            self._chunk_idxs[name] += ID_TURNOVER[name]
        self._sample_ids[name] = sample_id

        if name == "EEG":
            self._chunks[name][0, :] = np.copy(self._last_eeg_data)
        self._chunks[name] *= SCALE_FACTOR[name]
        self._enqueue_chunk(name)

    def _unknown_packet_warning(self, start_byte, packet):
        """Print if incoming byte ID is unknown."""
        warn("Unknown Ganglion packet byte ID: {}".format(start_byte))

    def _parse_message(self, start_byte, packet):
        """Parse a partial ASCII message."""
        if "messages" in self._streamer.subscriptions:
            self._chunks["messages"] += str(packet)
            if start_byte == 207:
                self._enqueue_chunk("messages")
                self._chunks["messages"][0] = ""

    def _parse_uncompressed(self, packet_id, packet):
        """Parse a raw uncompressed packet."""
        if bad_data_size(packet, 19, "uncompressed data"):
            return
        # 4 channels of 24bits
        self._last_eeg_data[:] = [int_from_24bits(packet[i:i + 3])
                                  for i in range(0, 12, 3)]
        # = np.array([chan_data], dtype=np.float32).T
        self._update_counts_and_enqueue("EEG", packet_id)

    def _update_data_with_deltas(self, packet_id, deltas):
        for delta_id in [0, 1]:
            # convert from packet to sample ID
            sample_id = (packet_id - 1) * 2 + delta_id + 1
            # 19bit packets hold deltas between two samples
            self._last_eeg_data += np.array(deltas[delta_id])
            self._update_counts_and_enqueue("EEG", sample_id)

    def _parse_compressed_19bit(self, packet_id, packet):
        """Parse a 19-bit compressed packet without accelerometer data."""
        if bad_data_size(packet, 19, "19-bit compressed data"):
            return

        packet_id -= 100
        # should get 2 by 4 arrays of uncompressed data
        deltas = decompress_deltas_19bit(packet)
        self._update_data_with_deltas(packet_id, deltas)

    def _parse_compressed_18bit(self, packet_id, packet):
        """ Dealing with "18-bit compression without Accelerometer" """
        if bad_data_size(packet, 19, "18-bit compressed data"):
            return

        # set appropriate accelerometer byte
        id_ones = packet_id % 10 - 1
        if id_ones in [0, 1, 2]:
            value = int8_from_byte(packet[18])
            self._chunks["accelerometer"][0, id_ones] = value
            if id_ones == 2:
                self._update_counts_and_enqueue("accelerometer",
                                                packet_id // 10)

        # deltas: should get 2 by 4 arrays of uncompressed data
        deltas = decompress_deltas_18bit(packet[:-1])
        self._update_data_with_deltas(packet_id, deltas)

    def _parse_impedance(self, packet_id, packet):
        """Parse impedance data.

        After turning on impedance checking, takes a few seconds to complete.
        """
        raise NotImplementedError  # until this is sorted out...

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
    """Parse packet deltas from 19-bit compression format."""
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
    """Parse packet deltas from 18-byte compression format."""
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
