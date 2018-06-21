"""Plot time series data streamed through a dummy LSL outlet.
"""

import ble2lsl
from ble2lsl.devices import muse2016
from wizardhat import acquire, plot


if __name__ == '__main__':
    dummy_outlet = ble2lsl.DummyStreamer(muse2016)
    streamer = acquire.LSLStreamer()
    plot.Lines(streamer.data)
