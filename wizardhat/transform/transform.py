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


class PSD(Transformer):
    """Calculate the power spectral density for time series data.

    TODO:
        * add method that gets power in specified frequency bands?
        * decide default behaviour wrt fft method.. Welch is good for visualization
          but not really for using powers as features
        * the frequencies are computed automatically right now based on the nyquist
          frequency, which can have unexpected results/makes vis hard
        * probably shouldn't be computing an fft for every single new sample
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

class Filter(Transformer):
    """General class for online data filtering with `scipy.signal`

     Expects a single data source (e.g. EEG) with consistent units.

     This parent class accepts filter designs specified by passing `a` and `b`, 
     the coefficient vectors of a digital IIR or FIR filter. The filter is then 
     applied recursively using `scipy.signal.lfilter()`. 
     
     For examples of acceptable `a` and `b` coefficients, see the functions at:
     https://docs.scipy.org/doc/scipy/reference/signal.html#matlab-style-iir-filter-design
    """

    def __init__(self, buffer_in, a, b):
        """Create a new `Filter` object
        
        Args:
            buffer_in (buffers.Buffer): Buffer managing data to be filtered.
            a (array_like): Denominator coefficient vector of IIR or FIR filter
            b (array_like): Numerator coefficient vector of IIR or FIR filter
        """
        self.buffer_in = buffer_in #temporarily necessary
        self.ch_names = buffer_in.ch_names
        self.sfreq = buffer_in.sfreq
        self.a = a
        self.b = b

        self.initialize_filter()
        self.similar_output()
        Transformer.__init__(self, buffer_in=buffer_in)

    def initialize_filter(self):
        """uses filter coefficients `a` and `b` to get initial filter 
        state `zi` and initialize recursive filter state `self._z`"""
        zi = spsig.lfilter_zi(self.b, self.a)
        self._z = [zi]*len(self.ch_names)

    def apply_filter(self):
        """applies the filter to all data channels in `self._new` and formats
        result for the `buffer_out.update()` method"""
        filtered_samples = [[]]*len(self.ch_names)
        
        for i, ch in enumerate(self.ch_names):
            # TODO: change this awkward implementation
            x = self._new[ch]
            filt, z = spsig.lfilter(self.b, self.a, x, zi=self._z[i])
            filtered_samples[i] = list(filt)
            self._z[i] = list(z)

        return [tuple(s) for s in zip(*filtered_samples)]

    def similar_output(self):
        """Called in `__init__` when `buffer_out` has same form as `buffer_in`.
        """
        self.buffer_out = copy.deepcopy(self.buffer_in)
        self.buffer_out.update_pipeline_metadata(self)
        self.buffer_out.update_pipeline_metadata(self.buffer_out)

    def _buffer_update_callback(self):
        """Called by `buffer_in` when new data is available."""
        self._new = self.buffer_in.last_samples
        timestamps = self._new['time']
        samples = self.apply_filter()

        self.buffer_out.update(timestamps,samples)

class Bandpass(Filter):
    """Highpass, lowpass, and bandpass filtering via a Butterworth filter.

     Expects a single data source (e.g. EEG) with consistent units.

     TODO: 
        * decide whether "Bandpass" is really the best name, maybe "Butterworth?"
        * determine most clear way to pass low/high arguments, maybe throw 
          an error if neither is passed
        * automatically select filter order with `spsig.buttord` 
        * self._create_filter() and self._parse_filter_type()?
    """
    
    def __init__(self, buffer_in, low=None, high=None, order=4):
        """Create a new `Bandpass` object
        
        Args:
            buffer_in (buffers.Buffer): Buffer managing data to be filtered.
            low (float): Lower passband value (Hz); if `None`, value passed for
                `high` is used for a low-pass filter
            high (float): Upper passband value (Hz); if `None`, value passed for
                `low` is used for a high-pass filter
            order (int): Order of the Butterworth filter. Determines the sharpness of
                passband cutoff.
                Default: `4`
                
        """
        self.create_filter(low,high,order,buffer_in.sfreq)
        Filter.__init__(self, buffer_in=buffer_in, a=self.a, b=self.b)

    def create_filter(self,low,high,order,sfreq):
        """computes nyquist frequency and gets filter coefficients `a` and `b`"""
        nyq = 0.5 * sfreq
        filter_type, critical_freq = self.parse_filter_type(low,high,nyq)

        self.b, self.a = spsig.butter(order,critical_freq,btype=filter_type)

    def parse_filter_type(self,low,high,nyq):
        """ parses low/high arguments, normalizes low/high cutoffs with nyquist 
        frequency,and returns passband type"""
        if low is None and high is None:
            raise Exception('You must provide at least one passband value')
        if low is None:
            filter_type = 'lowpass'
            critical_freq = [float(high)/nyq]
        if high is None:
            filter_type = 'highpass'
            critical_freq = [float(low)/nyq]
        if None not in [low, high]:
            filter_type = 'bandpass' 
            critical_freq = [float(low)/nyq, float(high)/nyq]

        return filter_type, critical_freq


class Notch(Filter):
    """2nd-order IIR digital notch filter that removes a narrow frequency band.

     Expects a single data source (e.g. EEG) with consistent units.
     """
    
    def __init__(self, buffer_in, notch_freq, q=30):
        """Create a new `Notch` object
        
        Args:
            buffer_in (buffers.Buffer): Buffer managing data to be filtered.
            notch_freq (float): frequency to remove (Hz)
            q (float): Dimensionless quality factor. Controls the notch
                bandwidth (higher q means a wider notch)
                Default: `30`
                
        """
        self.create_filter(notch_freq, q, buffer_in.sfreq)
        Filter.__init__(self, buffer_in=buffer_in, a=self.a, b=self.b)

    def create_filter(self, notch_freq, q, sfreq):
        """computes nyquist frequency, normalizes notch band, and gets 
        filter coefficients `a` and `b`"""
        nyq = 0.5 * sfreq
        norm_freq = notch_freq / nyq

        self.b, self.a = spsig.iirnotch(norm_freq, q)