#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""

import threading

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


class LSLBufferPlot(threading.Thread):

    def __init__(self, data_buffer, subsample=2, figsize=(15, 6)):
        self.data_buffer = data_buffer
        sns.set(style="whitegrid")
        fig, axes = plt.subplots(1, 1, figsize=figsize, sharex=True)
        axes.set_xlabel('Time (s)')
        axes.xaxis.grid(False)
        axes.set_ylim(-data_buffer.n_chan + 0.5, 0.5)
        ticks = np.arange(0, -data_buffer.n_chan, -1)
        axes.set_yticks(ticks)
        impedances = np.std(data_buffer.data, axis=0)
        ticks_labels = ['{} - {}'.format(data_buffer.ch_names[ch], impedances[ch])
                        for ch in range(data_buffer.n_chan)]
        axes.set_yticklabels(ticks_labels)
        sns.despine(left=True)

        lines = []
        for ch in range(data_buffer.n_chan):
            line, = axes.plot(data_buffer.times[::subsample],
                              data_buffer.data[::subsample, ch] - ch, lw=1)
            lines.append(line)
        self.fig, self.axes, self.lines = fig, axes, lines

        self.refresh_step = int(0.2 / (12/data_buffer.sfreq))


    def run(self):
        k = 0
        while True:
