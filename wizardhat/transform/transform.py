"""Applying arbitrary transformations/calculations to `Data` objects.
"""

from wizardhat.buffers import Spectra
import wizardhat.utils as utils

import copy

import mne
import numpy as np


class Transformer:
    """Base class for transforming data handled by `Buffer` objects.

    Attributes:
        buffer_in (buffers.Buffer): Input data buffer.
        buffer_out (buffers.Buffer): Output data buffer.
    """

    def __init__(self, buffer_in):
        self.buffer_in = buffer_in
        self.buffer_in.event_hook += self._buffer_update_callback

    def similar_output(self):
        """Called in `__init__` when `buffer_out` has same form as `buffer_in`.
        """
        self.buffer_out = copy.deepcopy(self.buffer_in)
        self.buffer_out.update_pipeline_metadata(self)
        self.buffer_out.update_pipeline_metadata(self.buffer_out)

    def _buffer_update_callback(self):
        """Called by `buffer_in` when new data is available to filter."""
        raise NotImplementedError()


class MNETransformer(Transformer):
    """Parent class for MNE-based data processing.

    Expects a single data source (e.g. EEG) with consistent units.
    """

    def __init__(self, buffer_in, sfreq, source_type='eeg', scaling=1E6,
                 montage='standard_1020'):
        """Construct an `MNETransformer` instance.

        Args:
            buffer_in (buffers.TimeSeries): Input time series data.
            sfreq (int): Nominal sampling frequency of the time series.
            source_type (str): Source of data.
                See MNE documentation for acceptable values.
            scaling (float): Conversion from input units to MNE units.
                That is, `mne_value = input_value / scaling`.
            montage (str): Arrangement of electrodes, for EEG data.
                See MNE documentation for acceptable montages.
        TODO:
            * sfreq from timestamps? (not nominal)
        """
        Transformer.__init__(self, buffer_in=buffer_in)

        channel_types = [source_type] * len(buffer_in.ch_names)
        self.source_type = source_type
        self.info = mne.create_info(buffer_in.ch_names, sfreq, channel_types)
        self._sfreq = sfreq

        if source_type == 'eeg':
            self.montage = mne.channels.read_montage(montage,
                                                     ch_names=buffer_in.ch_names)
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


class MNEFilter(MNETransformer):
    """Apply MNE filters to TimeSeries buffer objects."""

    def __init__(self, buffer_in, l_freq, h_freq, sfreq, update_interval=10):
        """Construct an `MNEFilter` instance.

        Args:
            buffer_in (buffers.TimeSeries): Input time series.
            l_freq (float): Low-frequency cutoff.
            h_freq (float): High-frequency cutoff.
            sfreq (int): Nominal sampling frequency of input.
            update_interval (int): How often (in terms of input updates) to
                filter the data.
        """
        MNETransformer.__init__(self, buffer_in=buffer_in, sfreq=sfreq)
        self.similar_output()

        self._band = (l_freq, h_freq)

        self._update_interval = update_interval
        self._count = 0

    def _buffer_update_callback(self):
        self._count += 1
        if self._count == self._update_interval:
            data = self.buffer_in.unstructured
            timestamps, samples = data[:, 1], data[:, 1:]
            filtered = mne.filter.filter_data(samples.T, self._sfreq,
                                              *self._band)
            # samples_mne = self._to_mne_array(samples)
            # filtered_mne = samples_mne.filter(*self._band)
            # filtered = self._from_mne_array(filtered_mne)
            self.buffer_out.update(timestamps, filtered.T)
            self._count = 0


class PSD(Transformer):
    def __init__(self, buffer_in, sfreq=256, n_samples=256, pow2=True,
                 window=np.hamming):
        Transformer.__init__(self, buffer_in=buffer_in)
        self.sfreq = sfreq
        self.n_fft = utils.next_pow2(n_samples)
        self.window = window(self.n_fft).reshape((self.n_fft, 1))
        self.indep_range = np.fft.rfftfreq(self.n_fft, 1 / self.sfreq)
        self.buffer_out = Spectra(self.buffer_in.ch_names, self.indep_range)

    def _buffer_update_callback(self):
        timestamp = self.buffer_in.last_sample["time"]
        data = self.buffer_in.unstructured[-self.n_fft:, :]
        psd = self._get_power_spectrum(data)
        self.buffer_out.update(timestamp, psd.T)

    def _get_windowed(self, data):
        data_centered = data - np.mean(data, axis = 0)
        data_windowed = data_centered * self.window
        return data_windowed

    def _get_power_spectrum(self, data):
        data_windowed = self._get_windowed(data)
        data_fft = np.fft.rfft(data_windowed, n=self.n_fft, axis=0)
        data_fft /= self.n_fft
        psd = 2 * np.abs(data_fft)
        return psd
