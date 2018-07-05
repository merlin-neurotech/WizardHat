"""Specification/abstract parent for BLE2LSL device files."""

import numpy as np


class PacketHandler:
    """Abstract parent for device-specific packet manager classes."""

    def __init__(self, device_params, output_queue, **kwargs):
        """Construct a `PacketHandler` instance.

        Args:
            device_params (dict): Device-specific parameters.
            output_queue (queue.Queue): Queue for putting processed data.
        """
        n_chan = len(device_params["ch_names"])
        self._data = np.zeros((n_chan, device_params["chunk_size"]))
        self._sample_idxs = np.zeros(n_chan)

        self._output_queue = output_queue

        try:
            self.scaling_output = kwargs["scaling_output"]
        except KeyError:
            self.scaling_output = True

    def process_packet(self, packet, handle):
        """BLE2LSL passes incoming BLE packets to this method for parsing."""
        raise NotImplementedError()

    @property
    def output(self):
        """BLE2LSL expects data to be returned to the queue in this format."""
        return (np.copy(self._sample_idxs), np.copy(self._data))
