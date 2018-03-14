#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""

import copy
import datetime
import json
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

    def __init__(self, record=False, metadata=None, filename=None,
                 data_dir='data', label=''):
        # thread control
        self._lock = threading.Lock()
        self.updated = threading.Event()

        # IO
        self.record = record
        if filename is None:
            filename = self._new_filename(data_dir, label)
        # make sure data directory exists
        makedir(filename)
        self.data_dir = data_dir
        self._filename = filename

        # metadata
        # initialize if necessary, and keep record of pipeline
        if metadata is None:
            metadata = {}
        try:
            metadata.setdefault('pipeline', [])
        except TypeError:
            raise TypeError("Metadata must be a dict")
        self.metadata = metadata
        self.update_pipeline_metadata(self)

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

    def update_pipeline_metadata(self, obj):
        # TODO: More detailed object information
        self.metadata['pipeline'].append(type(obj).__name__)
        # write metadata to .json with same name as data file
        self._write_metadata_to_file()

    def _write_metadata_to_file(self):
        try:
            metadata_json = json.dumps(self.metadata, indent=4)
        except TypeError:
            raise TypeError("JSON could not serialize metadata")
        with open(self._filename + '.json', 'w') as f:
            f.write(metadata_json)

    def _new_filename(self, data_dir='data', label=''):
        date = datetime.date.today().isoformat()
        classname = type(self).__name__
        if label:
            label += '_'

        filename = './{}/{}_{}_{}{{}}'.format(data_dir, date, classname, label)
        # incremental counter to prevent overwrites
        # (based on existence of metadata file)
        count = 0
        while os.path.exists(filename.format(count) + '.json'):
            count += 1
        filename = filename.format(count)

        return filename

    def __deepcopy__(self, memo):
        cls = self.__class__
        clone = cls.__new__(cls)
        memo[id(self)] = clone
        for k, v in self.__dict__.items():
            # thread lock objects cannot be copied directly
            mask = {'_lock': threading.Lock(), 'updated': threading.Event()}
            if k in mask:
                setattr(clone, k, mask[k])
            else:
                setattr(clone, k, copy.deepcopy(v, memo))
        return clone


class TimeSeries(Data):
    """
    Attributes:
        window (float): Number of seconds of most recent data to store.
            Approximate, due to floor conversion to number of samples.
    """

    def __init__(self, ch_names, sfreq, n_samples=2560, record=True,
                 metadata=None, filename=None, data_dir='data', label=''):
        Data.__init__(self, record=record, metadata=metadata,
                      filename=filename, data_dir=data_dir, label=label)

        self.dtype = np.dtype({'names': ["time"] + ch_names,
                               'formats': ['f8'] * (1 + len(ch_names))})
        self.n_samples = n_samples
        self.sfreq = sfreq
        self.initialize()

    @classmethod
    def with_window(cls, ch_names, sfreq, window=10, **kwargs):
        """Make an instance with a given window length."""
        n_samples = int(window * sfreq)
        return cls(ch_names, sfreq, n_samples, **kwargs)

    def initialize(self, n_samples=None):
        """Initialize stored samples to zeros."""
        if n_samples is not None:
            self.n_samples = n_samples
        with self._lock:
            self._data = np.zeros((self.n_samples,), dtype=self.dtype)
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

    def _append(self, new):
        with self._lock:
            self._data = push_rows(self._data, new)

    def _write_to_file(self):
        with self._lock:
            with open(self._filename + ".csv", 'a') as f:
                for row in self._data:
                    line = ','.join(str(n) for n in row)
                    f.write(line + '\n')

    def _format_samples(self, timestamps, samples):
        """Format data `numpy.ndarray` from timestamps and samples."""
        stacked = [(t,) + tuple(s) for t, s in zip(timestamps, samples)]
        return np.array(stacked, dtype=self.dtype)

    @property
    def ch_names(self):
        """Names of channels."""
        # Assumes 'time' is in first column
        return self.dtype.names[1:]

    @property
    def window(self):
        """Actual number of seconds stored.

        Not necessarily the same as the requested window size due to flooring
        to the nearest sample.
        """
        return self.n_samples / self.sfreq

    @property
    def samples(self):
        """Return copy of samples, without timestamps."""
        return self.data[self.ch_names]

    @property
    def last(self):
        """Last sample stored."""
        with self._lock:
            return np.copy(self._data[-1])


class Transformer(threading.Thread):

    def __init__(self, data_in):
        threading.Thread.__init__(self)
        self.data_in = data_in

    def similar_output(self):
        """Call in `__init__` when `data_out` has same form as `data_in`."""
        self.data_out = copy.deepcopy(self.data_in)
        self.data_out.update_pipeline_metadata(self)
        self.data_out.update_pipeline_metadata(self.data_out)

    def run(self):
        raise NotImplementedError()


class MNETransformer(Transformer):
    """Parent class for MNE-based data processing.

    Expects a single data source (e.g. EEG) with consistent units.
    """
    def __init__(self, data_in, source_type='eeg', scaling=1E6,
                 montage='standard_1020'):
        Transformer.__init__(self, data_in=data_in)

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

    def __init__(self, data_in, ica_samples=1024, ica_freq=64,
                 method='fastica', n_exclude=1, filter_=False,
                 montage='standard_1020', autostart=True, **kwargs):

        MNETransformer.__init__(self, data_in=data_in, montage=montage)

        # output is similar to input
        self.similar_output()
        # ... but could be longer depending on ica_samples
        n_samples = max(ica_samples, self.data_in.n_samples)
        self.data_out.initialize(n_samples)

        self.ica = ICA(n_components=len(self.data_out.ch_names), method=method,
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

                self.data_out.update(samples_cleaned)


class FFT(Transformer):

    def __init__(self):
        pass

    def run(self):
        pass


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


def push_rows(arr, rows):
    """Add `rows` to the end of `arr` without changing size of `arr`"""
    n = arr.shape[0]
    arr = np.concatenate([arr, rows], axis=0)
    return arr[-n:]


def makedir(filename):
    os.makedirs(filename[:filename.rindex(os.path.sep)], exist_ok=True)
