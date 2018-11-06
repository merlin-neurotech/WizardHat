"""Applying arbitrary transformations/calculations to `Data` objects.
"""

from wizardhat.buffers import Spectra
import wizardhat.utils as utils

import copy

import mne
import numpy as np
import scipy.signal as spsig


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
    """Apply MNE filters to `TimeSeries` buffer objects."""

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
    """Calculate the power spectral density for time series data.

    TODO:
        * control over update frequency?
    """

    def __init__(self, buffer_in, n_samples=256, pow2=True, window=np.hamming):
        self.sfreq = buffer_in.sfreq
        if pow2:
            n_samples = utils.next_pow2(n_samples)
        self.n_fft = n_samples
        self.window = window(self.n_fft).reshape((self.n_fft, 1))
        self.indep_range = np.fft.rfftfreq(self.n_fft, 1 / self.sfreq)
        self.buffer_out = Spectra(buffer_in.ch_names, self.indep_range)

        Transformer.__init__(self, buffer_in=buffer_in)

    def _buffer_update_callback(self):
        """Called by `buffer_in` when new data is available."""
        timestamp = self.buffer_in.last_sample["time"]
        data = self.buffer_in.get_unstructured(last_n=self.n_fft)
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


class Convolve(Transformer):
    """Convolve a time series of data.

    Currently only convolves across the sampling dimension (e.g. the rows in
    unstructured data returned by a `buffers.TimeSeries` object) of all
    channels, and assumes that all channels have the same shape (i.e. as
    returned by the `get_unstructured` method.)
    """

    def __init__(self, buffer_in, conv_arr, conv_mode='valid',
                 conv_method='direct'):
        """Create a new `Convolve` object.

        Args:
            buffer_in (buffers.Buffer): Buffer managing data to convolve.
            conv_arr (np.ndarray): Array to convolve data with.
                Should not be longer than `buffer_in.n_samples`.
            conv_mode (str): Mode for `scipy.signal.convolve`.
                Default: `'valid'`.
            conv_method (str): Method for `scipy.signal.convolve`.
                Default: `'direct'`. For many channels and very large
                convolution windows, it may be faster to use `'fft'`.
        """
        Transformer.__init__(self, buffer_in=buffer_in)
        self.similar_output()
        self.conv_mode = conv_mode
        self.conv_method = conv_method

        # expand convolution array across independent (non-sampling) dims
        ch_shape = self.buffer_in.unstructured.shape[1:]
        self.conv_arr = np.array(conv_arr).reshape([-1] + [1] * len(ch_shape))
        self._conv_n_edge = len(self.conv_arr) - 1

        if self.conv_mode == 'valid':
            self._timestamp_slice = slice(self._conv_n_edge,
                                          -self._conv_n_edge)
        else:
            raise NotImplementedError()

    def _buffer_update_callback(self):
        """Called by `buffer_in` when new data is available."""
        n_new = self.buffer_in.n_new
        last_n = max(n_new + 2 * self._conv_n_edge, self.buffer_in.n_samples)
        data = self.buffer_in.get_unstructured(last_n=last_n)
        timestamps = self.buffer_in.get_timestamps(last_n=last_n)
        data_conv = spsig.convolve(data, self.conv_arr, mode=self.conv_mode,
                                   method=self.conv_method)
        self.buffer_out.update(timestamps[self._timestamp_slice], data_conv)


class MovingAverage(Convolve):
    """Calculate a uniformly-weighted moving average over a data series."""

    def __init__(self, buffer_in, n_avg):
        conv_arr = np.array([1 / n_avg] * n_avg)
        Convolve.__init__(self, buffer_in=buffer_in, conv_arr=conv_arr)
