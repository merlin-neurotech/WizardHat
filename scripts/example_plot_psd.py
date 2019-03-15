"""Plot time series data streamed through a dummy LSL outlet.
"""

import ble2lsl
from ble2lsl.devices import muse2016
from wizardhat import acquire, plot, transform

device = muse2016
plot_stream = "EEG"

if __name__ == '__main__':
    dummy_outlet = ble2lsl.Dummy(device)
    receiver = acquire.Receiver()
    psd = transform.PSD(receiver[plot_stream])
    plot.Spectra(psd.buffer_out)
    #plot.Lines(receiver.buffers[plot_stream])
