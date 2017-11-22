#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""

import numpy as np


EEG_BANDS = {
    "theta": (4, 8),
    "low_alpha": (8, 10),
    "med_alpha": (9, 11),
    "hi_alpha": (10, 12),
    "low_beta": (12, 21),
    "hi_beta": (21, 30),
    "alpha": (8, 12),
    "beta": (12, 30),
}
"""dict: Default EEG bands, ex. for feature separation."""


def epoching(samples, samples_epoch, samples_overlap=0):
    n_samples, n_chan = samples.shape

    samples_shift = int(samples_epoch - samples_overlap)
    n_epochs = 1 + int(np.floor((n_samples - samples_epoch) / samples_shift))
    epoched_samples = n_epochs * samples_epoch

    samples_, remainder = np.split(samples, np.array([epoched_samples]))
    markers = samples_shift * np.arange(n_epochs + 1)
    epochs = np.zeros((samples_epoch, n_chan, n_epochs))
    for i_epoch in range(n_epochs):
        ind_epoch = slice(markers[i_epoch], markers[i_epoch] + samples_epoch)
        epochs[:, :, i_epoch] = samples[ind_epoch, :]

    return epochs, remainder


def calc_fft(samples, sfreq, real_out=True):
    n_samples, n_chan = samples.shape
    hamming_window = np.hamming(n_samples)
    samples_centered = samples - np.mean(samples, axis=0)
    samples_centered_hamming = (samples_centered.T * hamming_window).T
    n_fft = int(2**(1 + np.floor(np.log2(n_samples))))
    fft = np.fft.fft(samples_centered_hamming, n=n_fft, axis=0) / n_samples
    if real_out:
        fft = fft[0:n_fft//2, :]
    return fft


def calc_psd(samples, sfreq):
    fft = calc_fft(samples, sfreq)
    psd = 2 * np.abs(fft)
    return psd


def calc_feature_vector(samples, sfreq, bands=EEG_BANDS):
    psd = calc_psd(samples, sfreq)
    f = sfreq/2 * np.linspace(0, 1, len(psd))
    feature_vector = np.array([make_features(band, psd, f)
                               for band in bands.items()])
    return feature_vector


def make_features(band, psd, cutoff):
    freqs = band[1]  # get dict values, not key
    ind = np.where((cutoff >= freqs[0]) & (cutoff <= freqs[1]))
    return np.nan_to_num(np.mean(psd[ind, :]))


def calc_feature_matrix(epochs, sfreq):
    n_epochs = epochs.shape[2]
    feat = calc_feature_vector(epochs[:, :, 0], sfreq).T
    feature_matrix = np.zeros((n_epochs, feat.shape[0]))

    for i_epoch in range(n_epochs):
        feature_matrix[i_epoch, :] = \
            calc_feature_vector(epochs[:, :, i_epoch], sfreq).T
    return feature_matrix


def epoching_alt(samples, samples_epoch, samples_overlap=0):
    n_samples, n_chan = samples.shape
    n_epochs = n_samples // samples_epoch
    epoched_samples = n_epochs * samples_epoch
    data_, remainder = np.split(samples, np.array([epoched_samples]))

    ind_rows = np.arange(epoched_samples).reshape(samples_epoch, n_epochs).T
    ind_rows = np.ravel(np.repeat(ind_rows, n_chan, axis=0))
    ind_cols = np.tile(np.repeat(np.arange(n_chan), n_epochs), samples_epoch)

    epochs = data_[ind_rows, ind_cols].reshape(samples_epoch, n_chan, n_epochs)

    return epochs, remainder
