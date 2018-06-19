"""Plotting of data in `data.Data` objects."""

import math
import numpy as np
from bokeh.plotting import figure, output_file, show
from bokeh.models import ColumnDataSource



class Plotter():
    """Base class for plotting."""
    def __init__(self, data):
        """Construct a `Plotter` instance.

        Args:
            data (data.Data): Data object managing data to be plotted.
            plot_params (dict): Plot display parameters.
        """
        self.data = data
        output_file('WizardHat Plotter.html')


class Lines(Plotter):
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
            self.data.ch_names
            self.data = [self.data]
        except AttributeError:
            pass

        p = figrue()

        self.start()

    def start():

        self.reformat()
        self.self.draw()

    def reformat(self):
        data = {'time':self.data['time']}
        for chan in self.data.ch_names:
            data[chan] = self.data.data[chan]
        
        self.source = ColumnDataSource(data)
    
    def draw(self):
        for chan in self.data.ch_names:
            p.line(x='time',y=chan,source=source)

        show(p)


