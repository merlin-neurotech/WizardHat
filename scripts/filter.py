
"""Basic EEG filtering example using MNE."""

import ble2lsl
from ble2lsl.devices import muse2016
from wizardhat import acquire, transform, plot

from vispy import app

if __name__ == '__main__':
    # connect to Muse headset and stream samples through LSL outlet
    outlet_streamer = ble2lsl.BLEStreamer.from_device(muse2016)
    
    # automatically resolve and connect to LSL outlet and store samples in an
    # instance of wizardhat.data.TimeSeries (the attribute streamer.data),
    # which by default writes all samples to a data file
    streamer = acquire.LSLStreamer()
    # apply bandpass filter to data acquired by streamer
    bandpass = transform.MNEFilter(streamer.data, 1.0, 100.0, streamer.sfreq)
    # continuously plot the output of the bandpass filter
    filter_plot = plot.Lines([streamer.data, bandpass.data_out])
    #normal_plot = plot.Lines(streamer.data)
    app.run()
