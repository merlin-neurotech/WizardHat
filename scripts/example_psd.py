import ble2lsl
from ble2lsl.devices import muse2016
from wizardhat import acquire, plot, transform

import pylsl as lsl

device = muse2016
plot_stream = 'EEG'

if __name__ == '__main__':
    streamer = ble2lsl.Dummy(device)
    receiver = acquire.Receiver()
    psd_transformer = transform.PSD(receiver.buffers['EEG'], n_samples=256)
    psd_averaged = transform.MovingAverage(psd_transformer.buffer_out, n_avg=5)
    plotter = plot.Spectra(psd_averaged.buffer_out)
