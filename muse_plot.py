"""Plot time series data streamed through a dummy LSL outlet.
"""

import ble2lsl
from ble2lsl.devices import muse2016, ganglion
from wizardhat import acquire, plot

import pylsl as lsl

device = muse2016
plot_stream = 'ACC'

if __name__ == '__main__':
    streamer = ble2lsl.Streamer(device)
    lsl_streams = {stream.type(): stream for stream in lsl.resolve_streams()}
    inlet = lsl.StreamInlet(lsl_streams[plot_stream])
    acquirer = acquire.LSLStreamer(inlet=inlet)
    plot.Lines(acquirer.data)
