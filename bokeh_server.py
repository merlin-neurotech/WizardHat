"""Rough implementation of a standalone bokeh server. 

bokeh serve --show bokeh_server.py    <-- to run

Currently just grabs the most recent sample from Data.data every time the 
periodic callback executes. This is probably not the best way to do it, because 
the sampling rate is arbitrarily based on the value for add_periodic_callback(). 
For example, you can set the callback time to something faster than the sampling 
rate and you'll see that each value in streamer.data.data gets sampled a few 
times (starts to look like a step function). Right now there's no good way to 
check that we're not dropping samples when updating.

Also just two manually retrieved channels for now as a proof of concept, but
the gridplot method seems to work well for this.

TODO:
    * Implement Omri's new Plotter methods to generalize the source generation
    * Figure out sampling method- possibly using Data's self.updated attribute
        to trigger an update? Maybe we can update everything "in-place" because
        data.data already has a built-in window..
    * Automatically determine device name/set to title?
"""

from bokeh.layouts import row, column, gridplot
from bokeh.models import ColumnDataSource
from bokeh.plotting import curdoc, figure
from bokeh.driving import count

import ble2lsl
from ble2lsl.devices import muse2016
from wizardhat import acquire

# create dummy LSL outlet mimicking Muse 2016 info
dummy = ble2lsl.DummyStreamer(muse2016)

# connect LSL inlet to dummy outlet
streamer = acquire.LSLStreamer()

# init source
source = ColumnDataSource(dict(time=[], TP9=[], TP10=[]))

# init figures
p = figure(plot_height=250, tools="xpan,xwheel_zoom,xbox_zoom,reset", 
    x_axis_type=None, y_axis_location="right")
p.x_range.follow = "end" # always follows new data in source
p.x_range.follow_interval = 5 # in s (based on the actual value of the x in p.line)
p.x_range.range_padding = 0 # we can play with this stuff

p2 = figure(plot_height=250, tools="xpan,xwheel_zoom,xbox_zoom,reset", 
    x_axis_type=None, y_axis_location="right")
p2.x_range.follow = "end"
p2.x_range.follow_interval = 5
p2.x_range.range_padding = 0

# create line renders
p.line(x='time', y='TP9', alpha=0.8, line_width=2, color='blue', source=source)
p2.line(x='time', y='TP10', alpha=0.8, line_width=2, color='blue', source=source)

def update():
    new_sample = streamer.data.data[-1]
    new_data = dict(
        time= [new_sample['time']],
        TP9=[new_sample['TP9']],
        TP10=[new_sample['TP10']]
    )

    source.stream(new_data, 500)

curdoc().add_root(gridplot([[p], [p2]], toolbar_location="left", plot_width=1000))
curdoc().add_periodic_callback(update, 50) # in ms
curdoc().title = "Dummy EEG Stream"