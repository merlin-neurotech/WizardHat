"""

"""

import wizardhat.plot.shaders as shaders

import math

import numpy as np
from seaborn import color_palette
from vispy import gloo, app, visuals


class Plotter(app.Canvas):
    """ """
    def __init__(self, data_in, title='Plotter'):
        app.Canvas.__init__(self, title=title, keys='interactive')
        self.data_in = data_in
        # self.data = np.array([data[self.ch_names[0]],data[self.ch_names[1]], data[self.ch_names[2]], data[self.ch_names[3]], data[self.ch_names[4]]])
        # self.data = self.data.T


class Plotter2D(Plotter):
    def __init__(self, data, signal, scale=500):

        self.signal = signal  # either 'raw' or 'psd'

        # Number of cols and rows in the table.
        nrows = len(data.ch_names)
        ncols = 1

        m = nrows * ncols  # Number of signals.
        n = data.n_samples # Number of samples per signal.

        # Various signal amplitudes.
        amplitudes = np.zeros((m, n)).astype(np.float32)
        gamma = np.ones((m, n)).astype(np.float32)

        # Generate the signals as a (m, n) array.
        y = amplitudes

        color = color_palette("RdBu_r", nrows)
        color = np.repeat(color, n, axis=0).astype(np.float32)

        # Signal 2D index of each vertex (row and col) and x-index (sample index
        # within each signal).
        index = np.c_[np.repeat(np.repeat(np.arange(ncols), nrows), n),
                      np.repeat(np.tile(np.arange(nrows), ncols), n),
                      np.tile(np.arange(n), m)].astype(np.float32)

        self.program = gloo.Program(shaders.VERT_SHADER, shaders.FRAG_SHADER)
        self.program['a_position'] = y.reshape(-1, 1)
        self.program['a_color'] = color
        self.program['a_index'] = index
        self.program['u_scale'] = (1., 1.)
        self.program['u_size'] = (nrows, ncols)
        self.program['u_n'] = n

        # text
        self.font_size = 48.
        self.names = []
        self.quality = []
        for ii in range(self.n_chan):
            text = visuals.TextVisual(self.ch_names[ii], bold=True, color='white')
            self.names.append(text)
            text = visuals.TextVisual('', bold=True, color='white')
            self.quality.append(text)

        self.quality_colors = color_palette("RdYlGn", 11)[::-1]

        self.scale = scale

        self._timer = app.Timer('auto', connect=self.updatePlot, start=True)
        gloo.set_viewport(0, 0, *self.physical_size)
        gloo.set_state(clear_color='black', blend=True,
                       blend_func=('src_alpha', 'one_minus_src_alpha'))

        self.show()

    def on_key_press(self, event):
        # increase time scale
        if event.key.name in ['+', '-']:
            if event.key.name == '+':
                dx = -0.05
            else:
                dx = 0.05
            scale_x, scale_y = self.program['u_scale']
            scale_x_new, scale_y_new = (scale_x * math.exp(1.0*dx),
                                        scale_y * math.exp(0.0*dx))
            self.program['u_scale'] = (max(1, scale_x_new), max(1, scale_y_new))
            self.update()

    def on_mouse_wheel(self, event):
        dx = np.sign(event.delta[1]) * .05
        scale_x, scale_y = self.program['u_scale']
        scale_x_new, scale_y_new = (scale_x * math.exp(0.0*dx),
                                    scale_y * math.exp(2.0*dx))
        self.program['u_scale'] = (max(1, scale_x_new), max(0.01, scale_y_new))
        self.update()

    def updatePlot(self, data):
        if self.signal == 'raw':
            plot_data = (self.data - self.data.mean(axis=0)) / self.scale
        elif self.signal =='psd':
            #plot_data = Transformer.FFT(self.data) #Uncomment when transformer class is ready
            self.signal == 'raw'
            pass
        sd = np.std(plot_data[-int(self.sfreq):], axis=0)[::-1] * self.scale
        co = np.int32(np.tanh((sd - 30) / 15)*5 + 5)
        for ii in range(self.n_chan):
            self.quality[ii].text = '%.2f' % (sd[ii])
            self.quality[ii].color = self.quality_colors[co[ii]]
            self.quality[ii].font_size = 12 + co[ii]

            self.names[ii].font_size = 12 + co[ii]
            self.names[ii].color = self.quality_colors[co[ii]]

        self.program['a_position'].set_data(plot_data.T.ravel().astype(np.float32))
        self.update()

    def on_resize(self, event):
        # Set canvas viewport and reconfigure visual transforms to match.
        vp = (0, 0, self.physical_size[0], self.physical_size[1])
        self.context.set_viewport(*vp)

        for ii, t in enumerate(self.names):
            t.transforms.configure(canvas=self, viewport=vp)
            t.pos = (self.size[0] * 0.025, ((ii + 0.5)/self.n_chan) * self.size[1])

        for ii, t in enumerate(self.quality):
            t.transforms.configure(canvas=self, viewport=vp)
            t.pos = (self.size[0] * 0.975, ((ii + 0.5)/self.n_chan) * self.size[1])

    def on_draw(self, event):
        gloo.clear()
        gloo.set_viewport(0, 0, *self.physical_size)
        self.program.draw('line_strip')
        [t.draw() for t in self.names + self.quality]


#class signalPlotter(Plotter)
 #   def __init__(self)
  #  self.initialize()
#
 #   def initialize(self):
        #streams, figures, assign colours, choose scale
        #prepare to take samples, know how many samples dealing with
        #draw channel names
  #      pass

   # def _draw(self):
        #Redraw figures with new updated information from TimeSeries
        #show raw voltage values on graph
    #    pass

    #def _on_filter_click(self, event):
        #redirect data to a filter function
        #call draw function with new filtered data
     #   pass

    #def _on_mouse_wheel(self, event):
        #rescale data (change window size)
     #   pass

    #def _mark_data(self, event):
        #manual add of event marker to be saved for later?
        #must call data save function to add the marker to appropriate timestamp
     #   pass

#class psdPlotter(Plotter)
 #   def __init__(self,data)
  #  self.initialize():
   # pass
    #def initialize():
        #set up view of power spectrum streaming
        #need to know how many bands to show
        #set up sample recieving variables
     #   pass

   # def _draw():
        #redraw PSD spectrum when new samples come in

    #    pass

    #def _process_sample():
        #call sample preprocessing function to generate frequency domain data
     #   pass
