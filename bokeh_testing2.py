import ble2lsl
from ble2lsl.devices import muse2016
from wizardhat import acquire
import math
import numpy as np
from bokeh.plotting import figure, output_file, show, curdoc
from bokeh.models import ColumnDataSource
from bokeh.transform import jitter
from bokeh.layouts import row, column, gridplot
from bokeh.driving import count

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
dummy_outlet = ble2lsl.DummyStreamer(muse2016)
streamer = acquire.LSLStreamer()


class Lines():
    """Multiple (stacked) line plots.
    Expects a two-dimensional `data.Data` object (such as `TimeSeries`) where
    all columns after the first give the data used to plot individual lines.
    Multiple data sources may be given in a list, assuming they have the same
    form (number of channels and rows/samples); the user can cycle between
    plots of each data source with the 'D' key.
    """
    def __init__(self, data):
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

        try:
            self.ch_names
            self.data = [self.data]
        except AttributeError:
            pass
        
        self.data = data
        self.source = ColumnDataSource({'time':[], 'TP9':[], 'AF7':[],'AF8':[],'TP10':[]})#,'Right AUX':[]})
        self.start()
    def start(self):
        self.set_layout()
        self.run_server()
    
    def set_layout(self):
        self.p = figure(plot_height=100, tools="xpan,xwheel_zoom,xbox_zoom,reset", 
        x_axis_type=None, y_axis_location="right")
        self.p.x_range.follow = "end" # always follows new data in source
        self.p.x_range.follow_interval = 5 # in s (based on the actual value of the x in p.line)
        self.p.x_range.range_padding = 0 # we can play with this stuff

        self.p2 = figure(plot_height=100, tools="xpan,xwheel_zoom,xbox_zoom,reset", 
        x_axis_type=None, y_axis_location="right")
        self.p2.x_range.follow = "end" # always follows new data in source
        self.p2.x_range.follow_interval = 5 # in s (based on the actual value of the x in p.line)
        self.p2.x_range.range_padding = 0 # we can play with this stuff

        self.p3 = figure(plot_height=100, tools="xpan,xwheel_zoom,xbox_zoom,reset", 
        x_axis_type=None, y_axis_location="right")
        self.p3.x_range.follow = "end" # always follows new data in source
        self.p3.x_range.follow_interval = 5 # in s (based on the actual value of the x in p.line)
        self.p3.x_range.range_padding = 0 # we can play with this stuff

        self.p4 = figure(plot_height=100, tools="xpan,xwheel_zoom,xbox_zoom,reset", 
        x_axis_type=None, y_axis_location="right")
        self.p4.x_range.follow = "end" # always follows new data in source
        self.p4.x_range.follow_interval = 5 # in s (based on the actual value of the x in p.line)
        self.p4.x_range.range_padding = 0 # we can play with this stuff
        
        self.p.line(x='time', y='TP9', alpha=0.8, line_width=2, color='blue', source=self.source)
        self.p2.line(x='time', y='AF7', alpha=0.8, line_width=2, color='blue', source=self.source)
        self.p3.line(x='time', y='AF8', alpha=0.8, line_width=2, color='blue', source=self.source)
        self.p4.line(x='time', y='TP10', alpha=0.8, line_width=2, color='blue', source=self.source)
          
    
    def _reformat_new_sample(self,new_sample):
        data_dict = {'time':[new_sample['time']]}
        for chan in self.data.ch_names[0:4]: #Exclude Aux
            data_dict[chan] = [(new_sample[chan])]

        return data_dict

    def update(self):
        new_sample = self.data.data[-1]
        data_dict = self._reformat_new_sample(new_sample)
        #self.source = ColumnDataSource(data_dict) 
        self.source.stream(data_dict, 500)
    
    #def draw(self):
    #    y_range = self.data.ch_names
    #    p = figure(plot_width=800,y_range=y_range,plot_height=600, x_axis_type='datetime')
        
    #    for chan in self.data.ch_names:
    #        p.line(x='time',y=chan,source=self.source)
            
    #    show(p)
    
    def run_server(self):
        curdoc().add_root(gridplot([[self.p],[self.p2],[self.p3],[self.p4]], toolbar_location="left", plot_width=1000))
        curdoc().add_periodic_callback(self.update, 50) # in ms
        curdoc().title = "Dummy EEG Stream"

Lines(streamer.data)