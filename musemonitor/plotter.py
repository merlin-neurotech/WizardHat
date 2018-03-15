
from vispy import gloo, app, visuals

import numpy as np
import math
from seaborn import color_palette




VERT_SHADER = """
#version 120
// y coordinate of the position.
attribute float a_position;
// row, col, and time index.
attribute vec3 a_index;
varying vec3 v_index;
// 2D scaling factor (zooming).
uniform vec2 u_scale;
// Size of the table.
uniform vec2 u_size;
// Number of samples per signal.
uniform float u_n;
// Color.
attribute vec3 a_color;
varying vec4 v_color;
// Varying variables used for clipping in the fragment shader.
varying vec2 v_position;
varying vec4 v_ab;
void main() {
    float nrows = u_size.x;
    float ncols = u_size.y;
    // Compute the x coordinate from the time index.
    float x = -1 + 2*a_index.z / (u_n-1);
    vec2 position = vec2(x - (1 - 1 / u_scale.x), a_position);
    // Find the affine transformation for the subplots.
    vec2 a = vec2(1./ncols, 1./nrows)*.9;
    vec2 b = vec2(-1 + 2*(a_index.x+.5) / ncols,
                  -1 + 2*(a_index.y+.5) / nrows);
    // Apply the static subplot transformation + scaling.
    gl_Position = vec4(a*u_scale*position+b, 0.0, 1.0);
    v_color = vec4(a_color, 1.);
    v_index = a_index;
    // For clipping test in the fragment shader.
    v_position = gl_Position.xy;
    v_ab = vec4(a, b);
}
"""

FRAG_SHADER = """
#version 120
varying vec4 v_color;
varying vec3 v_index;
varying vec2 v_position;
varying vec4 v_ab;
void main() {
    gl_FragColor = v_color;
    // Discard the fragments between the signals (emulate glMultiDrawArrays).
    if ((fract(v_index.x) > 0.) || (fract(v_index.y) > 0.))
        discard;
    // Clipping test.
    vec2 test = abs((v_position.xy-v_ab.zw)/v_ab.xy);
    if ((test.x > 1))
        discard;
}
"""


class Plotter(app.Canvas):

    def __init__(self, data_in, title='Plotter'):
        app.Canvas.__init__(self, title=title, keys='interactive')
        self.data_in = data_in

class Plotter(app.Canvas):
    def __init__(self, data, signal, scale=500):

        self.signal = signal #either 'raw' or 'psd'

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


        self.program = gloo.Program(VERT_SHADER, FRAG_SHADER)
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

        # toggle filtering
        if event.key.name == 'D':
            if self.signal == 'raw':
                self.signal = 'psd'
            else:
                self.signal =='raw'

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

        self.data = np.array([data[self.ch_names[0]],data[self.ch_names[1]], data[self.ch_names[2]], data[self.ch_names[3]], data[self.ch_names[4]]])
        self.data = self.data.T
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
