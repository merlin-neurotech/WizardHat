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
    # "alpha": (8, 12),
    # "beta": (12, 30),
}
"""dict: Default EEG bands, ex. for feature separation."""


def dejitter_timestamps(timestamps, sfreq, last_time=None):
        """Convert timestamps to have regular sampling intervals.

        Args:
            timestamps (List[float]): A list of timestamps.
            sfreq (int): The sampling frequency.
            last_time (float): Time of the last sample preceding this set of
                timestamps. Defaults to `-1/sfreq`.
        """
        if last_time is None:
            last_time = -1/sfreq
        dejittered = np.arange(len(timestamps), dtype=np.float64)
        dejittered /= sfreq
        dejittered += last_time + 1/sfreq
        return dejittered


def epoching(samples, samples_epoch, samples_overlap=0):
    """Split samples into epochs of uniform size.

    Args:
        samples (numpy.ndarray):
        samples_epoch (int): Number of samples per epoch.
        samples_overlap (int, optional): Samples of overlap between adjacent
            epochs.
    """
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


def calc_fft(samples, real_out=True):
    """Calculate the FFT from an array of samples.

    Args:
        samples (numpy.ndarray): Time-domain samples.
        real_out (bool): Whether to return only the first half of the result,
            which is sufficient for the real-valued frequency domain.
    """
    n_samples, n_chan = samples.shape
    hamming_window = np.hamming(n_samples)
    samples_centered = samples - np.mean(samples, axis=0)
    samples_centered_hamming = (samples_centered.T * hamming_window).T
    n_fft = int(2**(1 + np.floor(np.log2(n_samples))))
    fft = np.fft.fft(samples_centered_hamming, n=n_fft, axis=0) / n_samples
    if real_out:
        fft = fft[0:n_fft//2, :]
    return fft


def calc_psd(samples):
    """Calculate the power spectral density from an array of samples.

    Args:
        samples (numpy.ndarray): Time-domain samples.
    """
    fft = calc_fft(samples)
    psd = 2 * np.abs(fft)
    return psd


def calc_psd_band_means(samples, sfreq, bands=EEG_BANDS):
    """Calculate band PSD means from an array of samples.

    Args:
        samples (numpy.ndarray): Time-domain samples.
        sfreq (int): Sampling frequency.
        bands (Dict[str, tuple]): Maps band names to band hi and lo bounds.
    """
    psd = calc_psd(samples)
    fft_freqs = sfreq/2 * np.linspace(0, 1, len(psd))
    feature_vector = np.ravel([psd_band_mean(psd, band, fft_freqs)
                               for band in bands.items()])
    return feature_vector


def psd_band_mean(psd, band, fft_freqs):
    """Calculate the PSD mean for a given band.

    Args:
        band (Dict[str, tuple]): Maps the band name to the
        psd (numpy.ndarray): The full power spectral density.
        fft_freqs (numpy.ndarray): FFT frequencies associated with the PSD.
    """
    bounds = band[1]  # get dict values, not key
    ind = np.where((fft_freqs >= bounds[0]) & (fft_freqs <= bounds[1]))
    return np.nan_to_num(np.mean(psd[ind, :], axis=1))


def feature_matrix(epochs, sfreq):
    """Calculate the feature matrix from a set of epochs."""
    n_epochs = epochs.shape[2]
    feat = calc_psd_band_means(epochs[:, :, 0], sfreq).T
    feature_matrix = np.zeros((n_epochs, feat.shape[0]))

    for i_epoch in range(n_epochs):
        feature_matrix[i_epoch, :] = \
            calc_psd_band_means(epochs[:, :, i_epoch], sfreq).T
    return feature_matrix
