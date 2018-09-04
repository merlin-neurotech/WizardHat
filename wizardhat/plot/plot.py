"""Plotting of data in `buffers.Buffer` objects.

Rough implementation of a standalone bokeh server.

Currently just grabs the most recent sample from Buffers.buffer every time the
periodic callback executes. This is probably not the best way to do it, because
the sampling rate is arbitrarily based on the value for
`add_periodic_callback()`. For example, you can set the callback time to
something faster than the sampling rate and you'll see that each value in
`buffer.data` gets sampled a few times (starts to look like a step
function). Right now there's no good way to check that we're not dropping
samples when updating.

Also just two manually retrieved channels for now as a proof of concept, but
the gridplot method seems to work well for this.

TODO:
    * Automatically determine device name/set to title?
"""

from functools import partial

from bokeh.layouts import row,gridplot, widgetbox
from bokeh.models.widgets import Button, RadioButtonGroup
from bokeh.models import ColumnDataSource
from bokeh.palettes import all_palettes as palettes
from bokeh.plotting import figure
from bokeh.server.server import Server
import numpy as np
from tornado import gen


class Plotter:
    """Base class for plotting."""

    def __init__(self, buffer, autostart=True):
        """Construct a `Plotter` instance.

        Args:
            buffer (buffers.Buffer): Data object managing data to be plotted.
            plot_params (dict): Plot display parameters.
        """
        self.buffer = buffer
        self.buffer.event_hook += self._buffer_update_callback
        # output_file('WizardHat Plotter.html')
        self.server = Server({'/': self._app_manager})
        #self.add_widgets()
        self.autostart = autostart

    def _buffer_update_callback(self):
        """Called by `buffer` when new data is available."""
        raise NotImplementedError()

    def add_widgets(self):
        self.stream_option = RadioButtonGroup(labels=['EEG', 'ACC', 'GYR'], active=0)
        self.filter_option = RadioButtonGroup(labels=['Low Pass', 'High Pass', 'Band Pass'], active=0)
        self.widget_box = widgetbox(self.stream_option,
                                    self.filter_option,
                                    width=300)

    def run_server(self):
        self.server.start()
        self.server.io_loop.add_callback(self.server.show, '/')
        self.server.io_loop.start()

    def _app_manager(self, curdoc):
        self._curdoc = curdoc
        self._set_layout()
        self._set_callbacks()

    def _set_callbacks(self):
        #self._curdoc.add_root(row(self.widget_box,
        #                          gridplot(self.plots, toolbar_location="left",
        #                                   plot_width=1000)))
        self._curdoc.add_root(gridplot(self.plots, toolbar_location="left",
                                       plot_width=1000))
        self._curdoc.title = "WizardHat"


class Lines(Plotter):
    """Multiple (stacked) line plots.

    Expects a two-dimensional `buffers.Buffer` object (such as `TimeSeries`) where
    all columns after the first give the data used to plot individual lines.
    Multiple data sources may be given in a list, assuming they have the same
    form (number of channels and rows/samples); the user can cycle between
    plots of each data source with the 'D' key.
    """

    def __init__(self, buffer, n_samples=5000, palette='Category10',
                 bgcolor="white", **kwargs):
        """Construct a `Lines` instance.
        Args:
            buffer (buffers.Buffer or List[buffers.Buffer]): Objects with data
                to be plotted. Multiple objects may be passed in a list, in
                which case the plot can cycle through plotting the data in
                each object by pressing 'd'. However, all data objects passed
                should have a similar form (e.g. `TimeSeries` with same number
                of rows/samples and channels).
            plot_params (dict): Plot display parameters.
        """

        super().__init__(buffer, **kwargs)

        # TODO: initialize with existing samples in self.buffer.data
        data_dict = {name: []  # [self.buffer.data[name][:n_samples]]
                     for name in self.buffer.dtype.names}
        self._source = ColumnDataSource(data_dict)
        self._n_samples = n_samples

        self._colors = palettes[palette][len(self.buffer.ch_names)]
        self._bgcolor = bgcolor

        if self.autostart:
            self.run_server()

    def _set_layout(self):
        self.plots = []
        for i, ch in enumerate(self.buffer.ch_names):
            p = figure(plot_height=100,
                       tools="xpan,xwheel_zoom,xbox_zoom,reset",
                       x_axis_type='datetime', y_axis_location="right")#,y_range=(-10,10))
            p.x_range.follow = "end"  # always follows new data in source
            p.x_range.follow_interval = 5  # in s
            p.x_range.range_padding = 0  # we can play with this stuff
            p.yaxis.axis_label = ch
            p.background_fill_color = self._bgcolor
            # p.background_fill_alpha = 0.5
            p.line(x='time', y=ch, alpha=0.8, line_width=2,
                   color=self._colors[i], source=self._source)
            self.plots.append([p])

    @gen.coroutine
    def _update(self, data_dict):
        self._source.stream(data_dict, self._n_samples)

    def _buffer_update_callback(self):
        #TODO Time delay of 1 second is necessary because there seems to be plotting issue related to server booting
        #time delay allows the server to boot before samples get sent to it.
        data_dict = {name: self.buffer.last_samples[name]
                     for name in self.buffer.dtype.names}
        try:  # don't freak out if IOLoop
            self._curdoc.add_next_tick_callback(partial(self._update,
                                                        data_dict))
        except AttributeError:
            pass


class Spectra(Plotter):
    """
    """

    def __init__(self, buffer, palette='Category10', bgcolor="white",
                 **kwargs):
        """
        Args:
            buffer (buffers.Buffer or List[buffers.Buffer]): Objects with data
                to be plotted. Multiple objects may be passed in a list, in
                which case the plot can cycle through plotting the data in
                each object by pressing 'd'. However, all data objects passed
                should have a similar form (e.g. `TimeSeries` with same number
                of rows/samples and channels).
            plot_params (dict): Plot display parameters.
        """

        super().__init__(buffer, **kwargs)

        # TODO: initialize with existing samples in self.buffer.data
        data_dict = {name: np.zeros_like(self.buffer.range)
                     for name in self.buffer.dtype.names if not name == 'time'}
        data_dict["range"] = self.buffer.range
        self._source = ColumnDataSource(data_dict)

        self._colors = palettes[palette][len(self.buffer.ch_names)]
        self._bgcolor = bgcolor

        if self.autostart:
            self.run_server()

    def _set_layout(self):
        self.plots = []
        for i, ch in enumerate(self.buffer.ch_names):
            p = figure(plot_height=100,
                       tools="xpan,xwheel_zoom,xbox_zoom,reset",
                       y_axis_location="right")
            p.yaxis.axis_label = ch
            p.background_fill_color = self._bgcolor
            p.line(x='range', y=ch, alpha=0.8, line_width=2,
                   color=self._colors[i], source=self._source)
            self.plots.append([p])

    @gen.coroutine
    def _update(self, data_dict):
        self._source.data = data_dict

    def _buffer_update_callback(self):
        last_samples = self.buffer.last_samples
        data_dict = {name: last_samples[name].T
                     for name in self.buffer.dtype.names if not name == 'time'}
        data_dict['range'] = self.buffer.range
        try:  # don't freak out if IOLoop
            self._curdoc.add_next_tick_callback(partial(self._update,
                                                        data_dict))
        except AttributeError:
            pass
