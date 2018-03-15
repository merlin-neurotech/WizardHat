"""

"""

import copy
import threading

import mne
from mne.processing import ICA


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
