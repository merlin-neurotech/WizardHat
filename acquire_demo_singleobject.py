

import ble2lsl
from ble2lsl.devices import muse2016
from wizardhat import acquire, plot


sources = muse2016.SOURCES  # contains names of all available data sources

# the user might do this to find out the available sources
print(sources)
#  > ['EEG', 'accelerometer', 'gyroscope', 'telemetry', 'status']

# for some future examples, let's say the user is interested in a given source
my_source = "EEG"

# subscribe to all data sources and push to separate LSL streams
streamer = ble2lsl.Streamer(muse2016, subscriptions=sources)

### this is where the differences begin

# automatically discovers all available LSL streams
# NOTE: the use of with_name in this way will not change the acquired streams
# if the streams created by the Streamer instance are the only ones on the
# network; however this shows how you could select only the streams coming
# from a certain type of device if others might also be available
acquirer = acquire.Acquirer(with_name=muse2016.NAME)

# at this point we can plot our source of interest
plot.Lines(acquirer.data[my_source])

# a source-keyed dict of data objects:
datas = acquirer.data

# if we want to stop acquisition from all LSL streams
acquirer.stop()
# or of just one stream
acquirer.stop(my_source)
# or perhaps of two
acquirer.stop(["accelerometer", "gyroscope"])
