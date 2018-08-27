import ble2lsl
from ble2lsl.devices import muse2016
from wizardhat import acquire, transform

import pylsl as lsl

device = muse2016
plot_stream = 'EEG'

if __name__ == '__main__':
    streamer = ble2lsl.Streamer(device)
    receiver = acquire.Receiver()
    PSD = transform.PSD(receiver.buffers['EEG'])