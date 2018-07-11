

import ble2lsl
from ble2lsl.devices import muse2016
from wizardhat import acquire, plot


sources = muse2016.SOURCE  # contains names of all available data sources

# the user might do this to find out the available sources
print(sources)
#  > ['EEG', 'accelerometer', 'gyroscope', 'telemetry', 'status']

# for some future examples, let's say the user is interested in a given source
my_source = "EEG"

# subscribe to all data sources and push to separate LSL sources
streamer = ble2lsl.Streamer(muse2016, subscriptions=sources)

### this is where the differences begin

# NOTE: the use of with_name in this way will not change the acquired streams
# if the streams created by the Streamer instance are the only ones on the
# network; however this shows how you could select only the streams coming
# from a certain type of device if others might also be available
inlets = acquire.get_lsl_inlets(with_name=muse2016.NAME)
acquirers = {name: acquire.Acquirer(inlet) for name, inlet in inlets.items()}
# NOTE: the "name" here is the stream name, which is a combination of the
# device name and the source name, e.g. "Muse-accelerometer"; might even change
# to include some more unique identifier in case multiple devices with the
# same name are used. In any case these aren't the source names the user wants
# to work with, so given the assumption that they are working with a single
# device, the acquirers could be named based on the source. Perhaps the user
# could be prompted to make a selection if multiple devices are available.
# Also note that this is still something that has to be dealt with in the
# single-Acquirer or Manager cases; it is not necessarily more of a problem to
# user-friendliness here.

# the preceding two lines could be replaced by a function like
acquirers = acquire.get_acquirers(with_name=muse2016.NAME)
# with the same result, a dict where each Acquirer instance is available like:
acquirers[my_source]
# where acquirers uses the source names familiar to the user as suggested above

# at this point we could plot our source of interest
plot.Lines(acquirers[my_source].data)

# we could get a source-keyed dict of data objects:
datas = {source: acquirers[source].data for source in sources}
# (we should probably renamed the Data class to Buffer or something)
# the same thing could be done by using a function like
datas = acquire.get_datas(acquirers)

# if we want to stop acquisition from all LSL streams
for source in acquirers:
    acquirers[source].stop()
# or of just one stream
acquirers["EEG"].stop()
# or perhaps of two
for source in ["accelerometer", "gyroscope"]:
    acquirers[source].stop()

# a function could also be made to make this less cumbersome
acquire.stop(acquirers)  # all
acquire.stop(acquirers, "EEG")  # one
acquire.stop(acquirers, ["accelerometer", "gyroscope"])  # two
