"""Specification/abstract parent for BLE2LSL device files.

To support a new BLE device, create a new module in `ble2lsl.devices` and
include the following module-level attributes:

    NAME (str): The device's name.
        Used for automatically finding the device address, so should be some
        substring of the name found by `pygatt`'s '`adapter.scan`.
    MANUFACTURER (str): The name of the device's manufacturer.
    STREAMS (List[str]): Names of data sources provided by the device.
    PARAMS (dict): Device-specific parameters, containing two dictionaries:
        streams (dict): Contains general per-stream parameters. Each member
            should be a dictionary with the names of the potential stream
            subscriptions as keys for the stream's respective value(s) of the
            parameter. (See `muse2016` module for an example of how to do this
            without being very messy.)

            type (dict[str]): The type of data to be streamed. Should be an
                `XDF` format string when possible.
            channel_count (dict[int]): The number of channels in the stream.
            nominal_srate (dict[float]): The stream's design sample rate (Hz).
                Used to generate dejittered timestamps for incoming samples.
            channel_format (dict[str]): The LSL datatype of the stream's data.
                LSL streams are of a single datatype, so one string should be
                given for each stream.
            numpy_dtype (dict[str or numpy.dtype]): The Numpy datatype for
                stream data.
                This will not always be identical to `channel_format`; for
                example, `'string'` is the string type in LSL but not in Numpy.
            units (dict[Iterable[str]]): Units for each channel in the stream.
            ch_names (dict[Iterable[str]]): Name of each channel in the stream.
            chunk_size (dict[int]): No. of samples pushed at once through LSL.

        ble (dict): Contains BLE-specific device parameters. Must contain
            keys for each of the streams named in `STREAMS`, with values of
            one or more characteristic UUIDs that `ble2lsl` must subscribe
            to when providing that stream. Some of these may be redundant or
            empty strings, as long as the device's `PacketHandler` separates
            incoming packets' data into respective streams (for example,
            see `ganglion`).

            address_type (BLEAddressType): One of `BLEAddressType.public` or
                `BLEAddressType.random`, depending on the device.
            interval_min (int): Minimum BLE connection interval.
            interval_max (int): Maximum BLE connection interval.
                Connection intervals are multiples of 1.25 ms. A good choice of
                `interval_min` and `interval_max` may be necessary to prevent
                dropped packets.
            send (str): UUID for the send/control characteristic.
                Control commands (e.g. to start streaming) are written to this
                characteristic.
            stream_on: Command to write to start streaming.
            stream_off: Command to write to end streaming.


As devices typically do not share a common format for the packets sent over
BLE, include a subclass of `PacketHandler` in the device file. This subclass
should provide a `process_packet` method, to which BLE2LSL will pass incoming
packets and the handles of the BLE characteristics from which they were
received. This method should perform any necessary processing on the packets,
delegating to other methods in the device file if necessary. After filling the
`_chunks` and `_chunk_idxs` attributes for a given stream, the chunk may be
enqueued for processing by `ble2lsl` by passing the stream name to
`_enqueue_chunk()`.

Summary of necessary inclusions to support a data source provided by a device:
    * A name for the stream in `STREAMS`.
    * Corresponding entries in each member of `PARAMS["streams"]`, and an entry
      in `PARAMS["ble"]` containing one or more UUIDs for characteristics
      that constitute a BLE subscription to the data source.
    * A means for `process_packet` to map an incoming packet to the appropriate
      stream name, typically using its handle; by passing this name with the
      enqueued chunk, this ensures it is pushed to the appropriate LSL stream
      by `ble2lsl.Streamer`.
    * Methods to process the contents of the packets and render them into
      appropriately-sized chunks as specified by the `channel_count` and
      `chunk_size` members of `PARAMS["streams"]`, and to return the chunks
      at the appropriate time (e.g. if multiple packets must be received to
      fill a single chunk.)

See `ble2lsl.devices.muse2016` for an example device implementation.

When a user instantiates `ble2lsl.Streamer`, they may provide a list
`DEFAULT_SUBSCRIPTIONS` of stream names to which to subscribe, which should be
some subset of the `STREAMS` attribute of the respective device file.

.. _XDF:
   https://github.com/sccn/xdf/wiki/Specifications
"""

from ble2lsl import empty_chunks, stream_idxs_zeros

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
        self._chunk_idxs = stream_idxs_zeros(subscriptions)

    def process_packet(self, handle, packet):
        """BLE2LSL passes incoming BLE packets to this method for parsing."""
        raise NotImplementedError()

    def _enqueue_chunk(self, name):
        """Ensure copies are returned."""
        self._transmit_queue.put((name,
                                  self._chunk_idxs[name],
                                  np.copy(self._chunks[name])
                                  ))
