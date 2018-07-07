"""Interfacing parameters for the Muse headband (2016 version)."""

from ble2lsl.devices.device import BasePacketHandler

import ast

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

# TODO: dict of packet formats
CH_PACKET_FORMAT = 'uint:16' + ',uint:12' * PARAMS["chunk_size"]
CH_HANDLE_IDXS = {32: 0, 35: 1, 38: 2, 41: 3, 44: 4}
CH_HANDLE_RECEIVE_ORDER = [44, 41, 38, 32, 35]

STATUS_PACKET_FORMAT = ','.join(['uint:8'] * 20)
STATUS_HANDLE = 14

TELEMETRY_PACKET_FORMAT = ','.join(['uint:16'] * 5)
TELEMETRY_HANDLE = 26

IMU_PACKET_FORMAT = 'uint:16' + ',int:16' * 9
ACCEL_HANDLE = 23
SCALE_FACTOR_ACCEL = 0.0000610352
GYRO_HANDLE = 20
SCALE_FACTOR_GYRO = 0.0074768


def convert_count_to_uvolts(value):
    return 0.48828125 * (value - 2048)


class PacketHandler(BasePacketHandler):
    """Process packets from the Muse 2016 headset into chunks.

    TODO:
        * Try callbacks instead of queues, and infer subscriptions
        * Timestamps for all subscriptions
        * better elif mechanism/generalization in process_packet: handle-method mapping
    """

    def __init__(self, output_queue, subscribes, **kwargs):
        super().__init__(device_params=PARAMS,
                         output_queue=output_queue,
                         **kwargs)
        self.subscribes = subscribes
        self._queues = {}
        self._message = ""

    def process_packet(self, packet, handle):
        """TODO: """
        if handle in CH_HANDLE_RECEIVE_ORDER:
            packet_idx, ch_values = self._unpack_channel(packet)
            idx = CH_HANDLE_IDXS[handle]
            self._data[idx] = ch_values
            self._sample_idxs[idx] = packet_idx
            if handle == CH_HANDLE_RECEIVE_ORDER[-1]:
                self._queues["channel"].put(self.output)
        elif handle == STATUS_HANDLE and self.subscribes["status"]:
            status_message_partial = self._unpack_status(packet)
            self._message += status_message_partial
            if status_message_partial[-1] == '}':
                self._message = self._message.replace('\n', '')
                # parse and enqueue dict
                self._queues["status"].put(ast.literal_eval(self._message))
                self._message = ""
        elif handle == TELEMETRY_HANDLE and self.subscribes["telemetry"]:
            telemetry = self._unpack_telemetry(packet)
            self._queues["telemetry"].put(telemetry)
        elif handle == ACCEL_HANDLE and self.subscribes["accelerometer"]:
            packet_index, samples = self._unpack_imu(packet)
            samples *= SCALE_FACTOR_ACCEL
            self._queues["accelerometer"].put((packet_index, samples))
        elif handle == GYRO_HANDLE and self.subscribes["gyroscope"]:
            packet_index, samples = self._unpack_imu(packet)
            samples *= SCALE_FACTOR_GYRO
            self._queues["gyroscope"].put((packet_index, samples))

    def _unpack_channel(self, packet):
        """Parse the bitstrings received over BLE."""
        unpacked = _unpack(packet, CH_PACKET_FORMAT)
        packet_index = unpacked[0]
        packet_values = np.array(unpacked[1:])
        if self.scaling_output:
            packet_values = convert_count_to_uvolts(packet_values)

        return packet_index, packet_values

    def _unpack_imu(self, packet):
        unpacked = _unpack(packet, IMU_PACKET_FORMAT)
        packet_index = unpacked[0]
        samples = np.array(unpacked[1]).reshape((3, 3))
        return packet_index, samples

    def _unpack_status(self, packet):
        unpacked = _unpack(packet, STATUS_PACKET_FORMAT)
        status_message = "".join(chr(i) for i in unpacked[1:])[:unpacked[0]]
        return status_message

    def _unpack_telemetry(self, packet):
        unpacked = _unpack(packet, TELEMETRY_PACKET_FORMAT)
        telemetry = {"battery": unpacked[1] / 512,
                     "fuel_gauge": unpacked[2] * 2.2,
                     "adc_volt": unpacked[3],
                     "temperature": unpacked[4]}
        return telemetry


def _unpack(packet, packet_format):
    packet_bits = bitstring.Bits(bytes=packet)
    unpacked = packet_bits.unpack(packet_format)
    return unpacked
