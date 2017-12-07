#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""

import threading

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


class LSLBufferPlot(threading.Thread):

    def __init__(self, data_buffer, subsample=2, figsize=(15, 6),
                 plot_params=None):
        self.data_buffer = data_buffer
        self.subsample = subsample
        if plot_params is None:
            plot_params = dict(style='whitegrid')
        n_chan = data_buffer.n_chan

        sns.set(**plot_params)
        fig, axes = plt.subplots(1, 1, figsize=figsize, sharex=True)
        axes.set_xlabel('Time (s)')
        axes.xaxis.grid(False)
        axes.set_ylim(-n_chan + 0.5, 0.5)
        ticks = np.arange(0, -n_chan, -1)
        axes.set_yticks(ticks)
        impedances = np.std(data_buffer.data, axis=0)
        line_labels = [format_line_label(ch) for ch in range(n_chan)]
        axes.set_yticklabels(line_labels)
        sns.despine(left=True)
        self.fig, self.axes = fig, axes

        self.lines = [init_line(ch) for ch in range(n_chan)]

        self.refresh_step = int(0.2 / (12/data_buffer.sfreq))

    def init_line(self, ch):
        line, = self.axes.plot(self.data_buffer.times[::self.subsample],
                               self.data_buffer.data[::self.subsample, ch] - ch,
                               lw=1)
        return line

    def format_line_label(self, ch):
        label = '{} - {}'.format(self.data_buffer.ch_names[ch],
                                 self.impedances[ch])
        return label

    def run(self):
        k = 0
        while True:
