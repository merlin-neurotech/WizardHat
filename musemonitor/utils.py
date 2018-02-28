#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""

import datetime
import os
import threading

import mne
from mne.preprocessing import ICA
import numpy as np


EEG_BANDS = {
    "delta": (0, 4),
    "theta": (4, 8),
    "low_alpha": (8, 10),
    #"med_alpha": (9, 11),
    "hi_alpha": (10, 12),
    "low_beta": (12, 21),
    "hi_beta": (21, 30),
    # "alpha": (8, 12),
    # "beta": (12, 30),
}
"""dict: Default EEG bands, ex. for feature separation."""


class Data:
    """Parent of data management classes.

    Attributes:
        metadata (dict):
    """

    def __init__(self, metadata=None):
        self._lock = threading.Lock()
        self.updated = threading.Event()
        self.metadata = metadata

    @property
    def data(self):
        """Return copy of data window."""
        try:
            with self._lock:
                return np.copy(self._data)
        except AttributeError:
            raise NotImplementedError()

    def initialize(self):
        """Initialize data window."""
        raise NotImplementedError()

    def update(self):
        """Update data."""
        raise NotImplementedError()


class TimeSeries(Data):
    """
    Attributes:
        window (float): Number of seconds of most recent data to store.
            Approximate, due to floor conversion to number of samples.
    """

    def __init__(self, ch_names, sfreq, window=10, record=True, metadata=None,
                 filename=None, data_dir='data', label=None):
        Data.__init__(self, metadata)
        names = ["time"] + ch_names
        self.dtype = np.dtype({'names': names,
                               'formats': ['f8'] * (1 + len(ch_names))})
        self.sfreq = sfreq
        self.n_samples = int(window * self.sfreq)
        self.initialize()

        if record:
            if filename is None:
                date = datetime.date.today().isoformat()
                filename = './{}/timeseries_{}_{{}}.csv'.format(data_dir, date)
                if label is None:
                    # use next available integer label
                    label = 0
                    while os.path.exists(filename.format(label)):
                        label += 1
                filename = filename.format(label)

            # make sure data directory exists
            os.makedirs(filename[:filename.rindex(os.path.sep)], exist_ok=True)

            self._file = open(filename, 'a')

    def initialize(self):
        """Initialize stored samples to zeros."""
        with self._lock:
            self._data = np.zeros((self.n_samples,), dtype=self.dtype)
            # self._data['time'] = np.arange(-self.window, 0, 1./self.sfreq)
        self._count = self.n_samples

    def update(self, timestamps, samples):
        """Append most recent chunk to stored data and retain window size."""
        new = self._format_samples(timestamps, samples)

        self._count -= len(new)

        #
        cutoff = len(new) + self._count
        self._append(new[:cutoff])
        if self._count < 1:
            self._write_to_file()
            self._append(new[cutoff:])
            self._count = self.n_samples

        self.updated.set()

    def _append(self, new):
        with self._lock:
            self._data = np.concatenate([self._data, new], axis=0)
            self._data = self.data[-self.n_samples:]

    def _write_to_file(self):
        with self._lock:
            for observation in self._data:
                line = ','.join(str(n) for n in observation) + '\n'
                self._file.write(line)

    def _format_samples(self, timestamps, samples):
        """Format data `numpy.ndarray` from timestamps and samples."""
        stacked = [(t,) + tuple(s) for t, s in zip(timestamps, samples)]
        return np.array(stacked, dtype=self.dtype)

    @property
    def ch_names(self):
        """Names of channels."""
        return self._data.dtype['channels'].names

    @property
    def window(self):
        """Actual number of seconds stored.

        Not necessarily the same as the requested window size due to flooring
        to the nearest sample.
        """
        return self.n_samples / self.sfreq


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
    samples_ = samples.view((samples.dtype[0], len(samples.dtype.names)))
    n_samples, n_chan = samples_.shape

    samples_shift = int(samples_epoch - samples_overlap)
    n_epochs = 1 + int(np.floor((n_samples - samples_epoch) / samples_shift))
    epoched_samples = n_epochs * samples_epoch

    samples_used, remainder = np.split(samples_, np.array([epoched_samples]))
    markers = samples_shift * np.arange(n_epochs + 1)
    epochs = np.zeros((samples_epoch, n_chan, n_epochs))
    for i_epoch in range(n_epochs):
        ind_epoch = slice(markers[i_epoch], markers[i_epoch] + samples_epoch)
        epochs[:, :, i_epoch] = samples_used[ind_epoch, :]

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


def calc_feature_matrix(epochs, sfreq):
    """Calculate the feature matrix from a set of epochs."""
    n_epochs = epochs.shape[2]
    feat = calc_psd_band_means(epochs[:, :, 0], sfreq).T
    feature_matrix = np.zeros((n_epochs, feat.shape[0]))

    for i_epoch in range(n_epochs):
        feature_matrix[i_epoch, :] = \
            calc_psd_band_means(epochs[:, :, i_epoch], sfreq).T
    return feature_matrix


def samples_threshold(samples, threshold):
    if np.mean(np.abs(samples)) > threshold:
        return True
    else:
        return False


class ICACleanup():
    def __init__(self, sfreq, ch_names, channel_types=None, filter_=False,
                 method='fastica', **kwargs):
        if channel_types is None:
            channel_types = ['eeg'] * len(ch_names)
        self.info = mne.create_info(ch_names, sfreq, channel_types)
        self.montage = mne.channels.read_montage('standard_1020',
                                                 ch_names=ch_names)
        self.ica = ICA(n_components=len(ch_names), method=method, **kwargs)
        self.picks = mne.pick_types(self.info, meg=False, eeg=True, eog=False)
        self.filter_ = filter_

    def remove_artifacts(self, samples, n_exclude=1, scaling=1E6):
        samples /= scaling
        samples_raw = mne.io.RawArray(samples.T, self.info);
        samples_raw.set_montage(self.montage)
        if self.filter_:
            samples_raw.filter(1., 100.)
        self.ica.fit(samples_raw, picks=self.picks);
        data_cleaned = self.ica.apply(samples_raw,
                                      exclude=list(range(n_exclude)))
        samples_cleaned, _ = data_cleaned[:]
        samples_cleaned *= scaling
        self.samples_raw = samples_raw
        return samples_cleaned

    @classmethod
    def from_lsl_streamer(cls, streamer, **kwargs):
        return cls(sfreq=streamer.sfreq, ch_names=streamer.ch_names, **kwargs)
