"""Plot time series data streamed through a dummy LSL outlet.
"""

import ble2lsl
from ble2lsl.devices import muse2016, ganglion
from wizardhat import acquire, plot

import pylsl as lsl

device = muse2016
plot_stream = 'EEG'

if __name__ == '__main__':
    streamer = ble2lsl.Streamer(device)
    receiver = acquire.Receiver()
    plot.Lines(receiver.buffers[plot_stream])
