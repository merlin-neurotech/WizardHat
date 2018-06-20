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
    * Figure out sampling method- possibly using Data's self.updated attribute
        to trigger an update? Maybe we can update everything "in-place" because
        data.data already has a built-in window..
    * Automatically determine device name/set to title?
    *Automatically determine source structure without explicit dictionary call
"""

import ble2lsl
from ble2lsl.devices import muse2016
from wizardhat import acquire

import math
from bokeh.driving import count
from bokeh.layouts import row, column, gridplot
from bokeh.models import ColumnDataSource
from bokeh.palettes import all_palettes as palettes
from bokeh.plotting import figure, output_file, show, curdoc
from bokeh.server.server import Server
from bokeh.transform import jitter

import numpy as np


class Lines():
    """Multiple (stacked) line plots.
    Expects a two-dimensional `data.Data` object (such as `TimeSeries`) where
    all columns after the first give the data used to plot individual lines.
    Multiple data sources may be given in a list, assuming they have the same
    form (number of channels and rows/samples); the user can cycle between
    plots of each data source with the 'D' key.
    """
    def __init__(self, data, palette='Category10', autostart=True):
        """Construct a `Lines` instance.
        Args:
            data (data.Data or List[data.Data]): Data object(s) managing data
                to be plotted. Multiple objects may be passed in a list, in
                which case the plot can cycle through plotting the data in
                each object by pressing 'd'. However, all data objects passed
                should have a similar form (e.g. `TimeSeries` with same number
                of rows/samples and channels).
            plot_params (dict): Plot display parameters.
        """

        self.data = data
        data_dict = {name: [] for name in self.data.dtype.names}
        self.source = ColumnDataSource(data_dict)

        self._colors = palettes[palette][len(self.data.ch_names)]

        if autostart:
            self.run_server()

    def _set_layout(self):
        self.p = []
        for i, ch in enumerate(self.data.ch_names):
            p = figure(plot_height=100,
                       tools="xpan,xwheel_zoom,xbox_zoom,reset",
                       x_axis_type=None, y_axis_location="right")
            p.x_range.follow = "end"  # always follows new data in source
            p.x_range.follow_interval = 5  # in s (based on the actual value of the x in p.line)
            p.x_range.range_padding = 0  # we can play with this stuff
            p.yaxis.axis_label = ch
            p.line(x='time', y=ch, alpha=0.8, line_width=2,
                   color=self._colors[i], source=self.source)
            self.p.append([p])

    def update(self):
        new_sample = self.data.data[-1]
        data_dict = {ch: [new_sample[ch]] for ch in self.data.dtype.names}
        self.source.stream(data_dict, 500)

    def _set_callbacks(self):
        self.curdoc.add_root(gridplot(self.p, toolbar_location="left",
                                      plot_width=1000))
        self.curdoc.add_periodic_callback(self.update, 50) # in ms
        self.curdoc.title = "Dummy EEG Stream"

    def _app_manager(self, curdoc):
        self.curdoc = curdoc
        self._set_layout()
        self._set_callbacks()

    def run_server(self):
        server = Server({'/': self._app_manager})
        server.start()
        server.io_loop.add_callback(server.show, '/')
        server.io_loop.start()


if __name__ == '__main__':
    dummy_outlet = ble2lsl.DummyStreamer(muse2016)
    streamer = acquire.LSLStreamer()
    Lines(streamer.data)
