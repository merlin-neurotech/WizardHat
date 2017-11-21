#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""

import numpy as np

def epoching_old(data, samples_epoch, samples_overlap=0):
    n_samples, n_chan = data['ch_values'].shape

    samples_shift = int(samples_epoch - samples_overlap)

    n_epochs = 1 + int(np.floor((n_samples - samples_epoch) / samples_shift))

    markers = samples_shift * np.arange(0, n_epochs + 1)

    epochs = np.zeros((samples_epoch, n_chan, n_epochs))
    for i_epoch in range(0,n_epochs):
        epochs[:,:,i_epoch] = data['ch_values'][ markers[i_epoch] : markers[i_epoch] + samples_epoch ,:]

    if (markers[-1] != n_samples):
        remainder = data['ch_values'][markers[-1] : n_samples, :]
    else:
        remainder = np.asarray([])

    return epochs , remainder

def epoching(data, samples_epoch, samples_overlap=0):
    n_epochs = n_samples // samples_epoch
    epoched_samples = n_epochs * samples_epoch

    epoch_rows = np.reshape(np.arange(0, epoched_samples),
                            (samples_epoch, n_epochs)).T
    rows = np.ravel(np.repeat(epoch_rows, n_chan, axis=0))
    cols = np.tile(np.repeat(np.arange(0, n_chan), n_epochs), samples_epoch)

    epochs = np.reshape(data[:epoched_samples][rows, cols],
                        (samples_epoch, n_chan, n_epochs))
    try:
        remainder = data[epoched_samples:]
    except IndexError:
        remainder = np.array([])
    return epochs, remainder
