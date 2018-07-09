"""Plot time series data streamed through a dummy LSL outlet.
"""

import ble2lsl
from ble2lsl.devices import muse2016
from wizardhat import acquire, plot

import pylsl as lsl

plot_stream = 'GYR'

if __name__ == '__main__':
    streamer = ble2lsl.Streamer(muse2016, subscriptions=["gyroscope"])
    lsl_streams = {stream.type(): stream for stream in lsl.resolve_streams()}
    inlet = lsl.StreamInlet(lsl_streams[plot_stream])
    acquirer = acquire.LSLStreamer(inlet=inlet)
    plot.Lines(acquirer.data)
