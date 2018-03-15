"""

"""

import utils

import copy
import datetime
import json
import os
import threading

import numpy as np


class Data:
    """Abstract base class of data management classes.

    Provides management of instance-related filenames and pipeline metadata for
    subclasses. Pipeline metadata consists of a field in the `metadata`
    attribute which tracks the `Data` and `transform.Transformer` subclasses
    through which the data has flowed. Complete instance metadata is written
    to a `.json` file with the same name as the instance's data file (minus
    its extension). Therefore, each data file corresponds to a `.json` file
    that describes how the data was generated.

    As an abstract class, `Data` should not be instantiated directly, but must
    be subclassed (e.g. `TimeSeries`). Subclasses should conform to expected
    behaviour by overriding methods or properties that raise
    `NotImplementedError`. A general implementation of the `data` property is
    provided, which will raise a `NotImplementedError` if the `_data` attribute
    is undefined; thus, instance data should be stored internally in
    `self._data`.

    Does not operate in a separate thread, but is accessed by separate threads.
    Subclasses should use the thread lock so that IO operations from multiple
    threads do not violate the data (e.g. adding a new row midway through
    returning a copy).

    Also provides an implementation of `__deepcopy__`, so that an independent
    but otherwise identical instance can be cloned from an existing instance
    using `copy.deepcopy`. This may be useful for `transform.Transformer`
    subclasses that want to output data in a similar form to their input.

    Args:
        metadata (dict): Arbitrary information added to instance's `.json`.
        filename (str): User-defined filename for saving instance (meta)data.
            By default, a name is generated based on the date, the class name
            of the instance, the user-defined label (if specified), and an
            incrementing integer to prevent overwrites. For example,
            "2018-03-01_TimeSeries_somelabel_0".
        data_dir (str): Directory for saving instance (meta)data.
            May be relative (e.g. "data" or "./data") or absolute.
            Defaults to "./data".
        label (str): User-defined addition to standard filename.

    Attributes:
        updated (threading.Event): Flag for threads waiting for data updates.
        filename (str): Final (generated or specified) filename for writing.
        metadata (dict): All metadata included in instance's `.json`.

    Todo:
        * Implement with abc.ABC (prevent instantiation of Data itself)
        * Detailed pipeline metadata: not only class names but attribute values
    """

    def __init__(self, metadata=None, filename=None, data_dir='./data',
                 label=''):

        # thread control
        self._lock = threading.Lock()
        self.updated = threading.Event()

        # IO
        if not data_dir[0] in ['.', '/']:
            data_dir = './' + data_dir
        if filename is None:
            filename = self._new_filename(data_dir, label)
        utils.makedir(filename)
        self.filename = filename

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
        """A complete copy of instance data.

        Copying prevents unwanted modification due to passing-by-reference.

        A general implementation is provided,
        """
        try:
            with self._lock:
                return np.copy(self._data)
        except AttributeError:
            raise NotImplementedError()
        except TypeError:
            pass

    def initialize(self):
        """Reset instance data; e.g. to zeros.

        May also contain other expressions necessary for initialization; for
        example, resetting the count of samples received.
        """
        raise NotImplementedError()

    def update(self):
        """Update instance data; e.g. by appending rows of new data."""
        raise NotImplementedError()

    def update_pipeline_metadata(self, obj):
        """Add some object's details to the instance's pipeline metadata.

        Automatically updates the instance's metadata `.json` file with the new
        information.

        Args:
            obj (object): The object to be represented in metadata.
        """
        self.metadata['pipeline'].append(type(obj).__name__)
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

        filename = '{}/{}_{}_{}{{}}'.format(data_dir, date, classname, label)
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
    """Manages 2D data consisting of rows of times and samples.

    Data is stored in a NumPy structured array where `'time'` is the first
    field and the remaining fields correspond to the channel names passed
    during instantiation.

    Args:
        ch_names (List[str]):
        sfreq (int):
        n_samples (int):
        record (bool):
        channel_formats (str or List[str]):

    Attributes:
        dtype (np.dtype):
        n_samples (int):
        sfreq (int):
    """

    def __init__(self, ch_names, sfreq, n_samples=2560, record=True,
                 channel_formats='f8', **kwargs):
        Data.__init__(self, **kwargs)

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
            self._data = utils.push_rows(self._data, new)

    def _write_to_file(self):
        with self._lock:
            with open(self.filename + ".csv", 'a') as f:
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
