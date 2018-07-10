"""Specification/abstract parent for BLE2LSL device files.

To support a new BLE device, create a new module in `ble2lsl.devices` and
include a `PARAMS` dictionary containing:
    manufacturer (str): The name of the device's manufacturer.
    name (str): The name of the device.
        Used for automatically finding the device address, so should be some
        substring of the name found by `pygatt`'s '`adapter.scan`.

    streams (dict): Contains stream-specific non-BLE parameters. Each member
        should be a dictionary with the names of the potential stream
        subscriptions as keys for the stream's respective value(s) of the
        parameter.

        type (dict[str]): The type of data to be streamed. Should be an `XDF`_
            format string when available.
        channel_count (dict[int]): The number of channels in the stream.
        nominal_srate (dict[float]): The stream's design sample rate (Hz).
            Used to generate regular timestamps for incoming samples.
        channel_format (dict[str]): The LSL datatype of the stream's data.
            LSL streams are of a single datatype, so one string should be given
            for each stream.
        numpy_dtype (dict[str or numpy.dtype]): The NumPy datatype of the data.
            This will not always be identical to `channel_format`; for example,
            `'string'` is the string type in LSL but not in NumPy.
        units (dict[Iterable[str]]): The units for each channel in the stream.
        ch_names (dict[Iterable[str]]): The name of each channel in the stream.
        chunk_size (dict[int]): No. of samples pushed at once through LSL.

    ble (dict): Contains BLE-specific parameters.
        address_type (BLEAddressType): One of `BLEAddressType.public` or
            `BLEAddressType.random`, depending on the device.
        interval_min (int): Minimum BLE connection interval.
        interval_max (int): Maximum BLE connection interval.
            Connection intervals are multiples of 1.25 ms.
        send (str): UUID for the send/control characteristic.
            Control commands (e.g. to start streaming) are written to this
            characteristic.
        stream_on: Command to write to start streaming.
        stream_off: Command to write to end streaming.
        disconnect (str, optional): UUID for the disconnect characteristic.


As devices typically do not share a common format for the packets sent over
BLE, include a subclass of `PacketHandler` in the device file. This subclass
should provide a `process_packet` method, to which BLE2LSL will pass incoming
packets and the handles of the BLE characteristics from which they were
received. This method should perform any necessary processing (delegating to
other methods in the device file if necessary) and pass the stream name, array
of sample IDs, and array of chunk data to the callback function passed to
`PacketHandler` during its instantiation by `ble2lsl.Streamer`.

To provide an LSL stream for a data stream provided by the device, the
following should be implemented:
    * A name for the LSL stream/data source with corresponding entries in
      each member of `PARAMS["streams"]`, and an entry with the same name
      in `PARAMS["ble"]` containing one or more UUIDs for characteristics
      that constitute a BLE subscription to the data source. A list of the
      available stream names must be provided in the `STREAMS` attribute.
    * A means for `process_packet` to map an incoming packet to the appropriate
      stream name, typically using its handle; by passing this name to the
      callback this ensures a chunk is pushed to the appropriate LSL stream by
      `ble2lsl.Streamer`.
    * Methods to process the contents of the packets and render them into
      appropriately-sized chunks as specified by the `channel_count` and
      `chunk_size` members of `PARAMS["streams"]`, and to return the chunks
      at the appropriate time (e.g. if multiple packets must be received to
      fill a single chunk.)

See `ble2lsl.devices.muse2016` for an example device implementation.

When a user instantiates `ble2lsl.Streamer`, they may provide a list of stream
names to which to subscribe, which should be some subset of the `STREAMS`
attribute of the respective device file.

.. _XDF:
   https://github.com/sccn/xdf/wiki/Specifications
"""

from ble2lsl import empty_chunks, empty_chunk_timestamps

import numpy as np


class BasePacketHandler:
    """Abstract parent for device-specific packet manager classes."""

    def __init__(self, stream_params, streamer, **kwargs):
        """Construct a `PacketHandler` instance.

        Args:
            stream_params (dict): Stream parameters.
                Pass `PARAMS["streams"]` from the device file.
            streamer (ble2lsl.Streamer): The master `Streamer` instance.
        """
        self._streamer = streamer
        self._transmit_queue = streamer._transmit_queue

        subscriptions = self._streamer.subscriptions
        self._chunks = empty_chunks(stream_params, subscriptions)
        self._sample_idxs = empty_chunk_timestamps(stream_params,
                                                   subscriptions,
                                                   dtype=int)

    def process_packet(self, handle, packet):
        """BLE2LSL passes incoming BLE packets to this method for parsing."""
        raise NotImplementedError()

    def _enqueue_chunk(self, name):
        """Ensure copies are returned."""
        self._transmit_queue.put((name,
                                  np.copy(self._sample_idxs[name]),
                                  np.copy(self._chunks[name])
                                  ))
