"""Specification/abstract parent for BLE2LSL device files.

To support a new BLE device, create a new module in `ble2lsl.devices` and
include a `PARAMS` dictionary containing:
    manufacturer (str): The name of the manufacturer.
    units (str): The units
    ch_names (Iterable[str]): The label for each channel in processed data.
    chunk_size (int): The number of samples per chunk passed through BLE2LSL.
    ble (dict): Contains BLE-specific parameters:
        address_type (BLEAddressType):
        service (str, optional):
        interval_min (int): Minimum BLE connection interval.
        interval_max (int): Maximum BLE connection interval.
        receive (List[str]): List of UUIDs for receive characteristics.
        send (str): UUID for the send characteristic.
        stream_on: Value to write to the send characteristic to start streaming.
        stream_off: Value to write to the send characteristic to end streaming.
        disconnect (str, optional): UUID for the disconnect characteristic.

Also include an `LSL_INFO` dictionary, which is used to construct the LSL
`StreamInfo` instance with outlet metadata. It must contain the keys:
    name (str): The device name, used for labels and for automatically finding
        the device address.
    type (str): The type of data to be streamed.
    channel_count (int): The number of channels of streaming data.
        Must be equal to the length of `PARAM["ch_names"]`.
    nominal_srate (float): The expected rate of sample streaming (Hz).
        Used to generate regular timestamps for incoming samples.
    channel_fmt (str): The datatype (e.g. "float32") of the streaming data.

As devices typically do not share a common format for the packets sent over
BLE, include a subclass of `PacketHandler` in the device file. This subclass
should provide a `process_packet` method, to which BLE2LSL will pass incoming
packets and the handles of the BLE characteristics from which they were
received. This method should perform any necessary processing (delegating to
other methods in the device file if necessary) but ultimately should put
processed tuples of data into the `queue.Queue` instance that will be passed
to the subclass's constructor as the `output_queue` argument. The contents of
these tuples should be as described in the abstract class in this module.
"""

import numpy as np


class BasePacketHandler:
    """Abstract parent for device-specific packet manager classes."""

    def __init__(self, device_params, callback, subscriptions,
                 **kwargs):
        """Construct a `PacketHandler` instance.

        Args:
            device_params (dict): Device-specific parameters.
            output_queue (queue.Queue): Queue for putting processed data.
        """
        self._chunks = {}
        self._sample_idxs = {}
        for name in subscriptions:
            channel_count = device_params["channel_count"][name]
            channel_format = device_params["channel_format"][name]
            self._chunks[name] = np.zeros((channel_count,
                                           device_params["chunk_size"][name]),
                                          dtype=channel_format)
            self._sample_idxs[name] = np.zeros(channel_count)

        self._callback = callback
        self._subscriptions = subscriptions

    def process_packet(self, packet, handle):
        """BLE2LSL passes incoming BLE packets to this method for parsing."""
        raise NotImplementedError()
