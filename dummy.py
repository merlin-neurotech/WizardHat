
"""Use of `ble2lsl.Dummy` to mimic a device.

The dummy LSL outlet is passed random samples, but mimics the number of
channels, nominal sample rate, and metadata for the device (Muse 2016
in this case).
"""

import ble2lsl
from ble2lsl.devices import muse2016
from wizardhat import acquire, plot
from vispy import app

if __name__ == '__main__':
    # create dummy LSL outlet mimicking Muse 2016 info
    dummy = ble2lsl.Dummy(muse2016)

    # connect LSL inlet to dummy outlet
    streamer = acquire.LSLStreamer()
    # plot dummy data
    dummy_plot = plot.Lines(streamer.data)
    # start vispy app
    app.run()

    
    
