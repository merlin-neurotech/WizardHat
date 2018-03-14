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

    def _write_to_file(self):
        with self._lock:
            for row in self._data:
                line = ','.join(str(n) for n in row) + '\n'
                self._file.write(line)


class TimeSeries(Data):
    """
    Attributes:
        window (float): Number of seconds of most recent data to store.
            Approximate, due to floor conversion to number of samples.
    """

    def __init__(self, ch_names, sfreq, n_samples, record=False, metadata=None,
                 filename=None, data_dir='data', label=None):
        Data.__init__(self, metadata)

        self.dtype = np.dtype({'names': ["time"] + ch_names,
                               'formats': ['f8'] * (1 + len(ch_names))})
        self.sfreq = sfreq
        self.initialize()

        self.record = record
        if self.record:
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
            self.data_dir = data_dir
            self._filename = filename
            self._file = open(filename, 'a')

    @classmethod
    def with_window(cls, ch_names, sfreq, window=10, **kwargs):
        n_samples = int(window * sfreq)
        return cls(ch_names, sfreq, n_samples, **kwargs)


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

        cutoff = len(new) + self._count
        self._append(new[:cutoff])
        if self._count < 1:
            if self.record:
                self._write_to_file()
            self._append(new[cutoff:])
            self._count = self.n_samples

        self.updated.set()

        #print(self._data)

    def _append(self, new):
        with self._lock:
            self._data = np.concatenate([self._data, new], axis=0)
            self._data = self._data[-self.n_samples:]

    def _write_to_file(self):
        with self._lock:
            #print('Saving File')
            self._file = open(self._filename, 'a')
            for observation in self._data:
                line = ','.join(str(n) for n in observation) + '\n'
                self._file.write(line)
            self._file.close()
            #print('File Saved')

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

    @property
    def last(self):
        """Last sample stored."""
        with self._lock:
            return np.copy(self._data[-1])


class Transformer(threading.Thread):

    def __init__(self, data_in):
        self.data_in = input_data

    def run(self):
        raise NotImplementedError()


class MNETransformer(Transformer):
    """Parent class for MNE-based data processing.

    Expects a single data source (e.g. EEG) with consistent units.
    """
    def __init__(self, data_in, source_type='eeg', scaling=1E6,
                 montage='standard_1020'):
        Transformer.__init__(data_in)

        channel_types = [source_type] * len(data_in.ch_names)
        self.source_type = source_type
        self.info = mne.create_info(data_in.ch_names, data_in.sfreq,
                                    channel_types)

        if source_type == 'eeg':
            self.montage = mne.channels.read_montage(montage,
                                                     ch_names=data_in.ch_names)
        if not source_type == 'meg':
            # MNE defaults to `meg=True` and everything else `False`...
            self.picks = mne.pick_types(self.info, meg=False,
                                        **{source_type: True})
        else:
            self.picks = mne.pick_types(self.info)

        self.scaling = scaling

    def _to_mne_array(self, samples):
        samples /= self.scaling
        mne_array = mne.io.RawArray(samples.T, self.info)
        if self.source_type == 'eeg':
            mne_array.set_montage(self.montage)
        return mne_array

    def _from_mne_array(self, mne_array):
        samples, _ = mne_array[:]
        samples *= self.scaling
        return samples


class ICAClean(MNETransformer):

    def __init__(self, data_in, ica_samples=1024, method='fastica',
                 n_exclude=1, filter_=False, autostart=True,
                 montage='standard_1020', **kwargs):
        MNETransformer.__init__(data_in, montage=montage)

        n_samples = max(data_in.n_samples, ica_samples)

        # TODO: better Data object copying?
        self.data = TimeSeries(ch_names=data_in.ch_names,
                               sfreq=data_in.sfreq,
                               n_samples=n_samples,
                               record=data_in.record,
                               metadata=data_in.metadata,
                               filename=None,
                               data_dir=data_in.data_dir,
                               label=None)
        self.ica = ICA(n_components=len(self.data.ch_names), method=method,
                       **kwargs)
        self.filter_ = filter_
        self.n_exclude = n_exclude

        self.proceed = True
        if autostart:
            self.start()

    def run(self):
        excludes = list(range(self.n_exclude))
        while self.proceed:
            if True: #TODO: count condition
                # TODO: exclude 'time': only EEG channels
                samples_mne = self._to_mne_array(self.data_in.data)
                if self.filter_:
                    samples_mne.filter(1.0, 100.0)
                self.ica_fit(samples_mne, picks=self.picks)

                samples_mne_cleaned = self.ica.apply(samples_mne,
                                                     exclude=excludes)
                samples_cleaned = self._from_mne_array(samples_mne_cleaned)

                self.data.update(samples_cleaned)


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
