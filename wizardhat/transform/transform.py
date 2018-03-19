"""Applying arbitrary transformations/calculations to `Data` objects.


TODO:
    * Switch from threading.Thread to multiprocessing.Process (not a good idea
      to use threads for CPU-intensive stuff)
"""

import copy
import threading

import mne
import numpy as np


class Transformer(threading.Thread):
    """Base class for transforming data stored in `Data` objects.

    Attributes:
        data_in (data.Data): Input data.
        data_out (data.Data): Output data.
    """
    def __init__(self, data_in):
        threading.Thread.__init__(self)
        self.data_in = data_in

    def similar_output(self):
        """Called in `__init__` when `data_out` has same form as `data_in`."""
        self.data_out = copy.deepcopy(self.data_in)
        self.data_out.update_pipeline_metadata(self)
        self.data_out.update_pipeline_metadata(self.data_out)

    def run(self):
        raise NotImplementedError()


class MNETransformer(Transformer):
    """Parent class for MNE-based data processing.

    Expects a single data source (e.g. EEG) with consistent units.
    """
    def __init__(self, data_in, sfreq, source_type='eeg', scaling=1E6,
                 montage='standard_1020'):
        """Construct an `MNETransformer` instance.

        Args:
            data_in (data.TimeSeries): Input time series data.
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
        Transformer.__init__(self, data_in=data_in)

        channel_types = [source_type] * len(data_in.ch_names)
        self.source_type = source_type
        self.info = mne.create_info(data_in.ch_names, sfreq, channel_types)
        self._sfreq = sfreq

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


class MNEFilter(MNETransformer):
    """Apply MNE filters to TimeSeries data objects."""

    def __init__(self, data_in, l_freq, h_freq, sfreq, update_interval=10):
        """Construct an `MNEFilter` instance.

        Args:
            data_in (data.TimeSeries): Input time series.
            l_freq (float): Low-frequency cutoff.
            h_freq (float): High-frequency cutoff.
            sfreq (int): Nominal sampling frequency of input.
            update_interval (int): How often (in terms of input updates) to
                filter the data.
        """
        MNETransformer.__init__(self, data_in=data_in, sfreq=sfreq)
        self.similar_output()

        self._band = (l_freq, h_freq)

        self._update_interval = update_interval
        self._count = 0
        self._proceed = True
        self.start()

    def run(self):
        # wait until data_in is updated
        while self._proceed:
            self.data_in.updated.wait()
            self.data_in.updated.clear()
            self._count += 1
            if self._count == self._update_interval:
                data = self.data_in.unstructured
                timestamps, samples = data[:, 1], data[:, 1:]
                filtered = mne.filter.filter_data(samples.T, self._sfreq,
                                                  *self._band)
                #samples_mne = self._to_mne_array(samples)
                #filtered_mne = samples_mne.filter(*self._band)
                #filtered = self._from_mne_array(filtered_mne)
                self.data_out.update(timestamps, filtered.T)
                self._count = 0

    def stop(self):
        self._proceed = False
