"""Objects for data storage.

All instantiable classes in this module are subclasses of the abstract class
`Buffer`, which enforces a common interface to its children. However, it imposes
no constraints on the structure of the stored data. Thus, storage in any kind
of data structure may be implemented in a subclass, so long as the appropriate
interface methods are defined in that subclass.

TODO:
    * Support for commonly-used data formats; HDF5 for longer recordings. XDF?
      MessagePack. JSON.
"""

import wizardhat.utils as utils

import atexit
import json
import os
import threading
import time

import numpy as np


class Buffer:
    """Abstract base class of data management classes.

    Provides management of instance-related filenames and pipeline metadata for
    subclasses. Pipeline metadata consists of a field in the `metadata`
    attribute which tracks the `Buffer` and `transform.Transformer` subclasses
    through which the data has flowed. Complete instance metadata is written
    to a `.json` file with the same name as the instance's data file (minus
    its extension). Therefore, each data file corresponds to a `.json` file
    that describes how the data was generated.

    As an abstract class, `Buffer` should not be instantiated directly, but must
    be subclassed (e.g. `TimeSeries`). Subclasses should conform to expected
    behaviour by overriding methods or properties that raise
    `NotImplementedError`; though `data` need not be overrided so long as the
    subclass's data structure is assigned to `_data`.

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
        filename (str): Final (generated or specified) filename for writing.
        metadata (dict): All metadata included in instance's `.json`.

    Todo:
        * Implement with abc.ABC (prevent instantiation of Buffer itself)
        * Detailed pipeline metadata: not only class names but attribute values
        * Decorator for locked methods
        * Is update really part of interface (Transformer expecting one type of
          Buffer will fail if it tries to update another, probably)
    """

    def __init__(self, metadata=None, filename=None, data_dir='./data',
                 label=''):

        # thread control
        self._lock = threading.Lock()
        self.event_hook = utils.EventHook()

        # file output preparations
        if not data_dir[0] in ['.', '/']:
            data_dir = './' + data_dir
        if filename is None:
            filename = self._new_filename(data_dir, label)
        utils.makedirs(filename)
        self.filename = filename
        self._data_dir = data_dir
        self._label = label

        # metadata
        if metadata is None:
            metadata = {}
        try:
            # initialize if necessary
            metadata.setdefault('pipeline', [])
        except TypeError:
            raise TypeError("Metadata must be a dict")
        self.metadata = metadata
        # add subclass information to pipeline metadata and write to file
        self.update_pipeline_metadata(self)

    @property
    def data(self):
        """A complete copy of instance data.

        Copying prevents unwanted modification due to passing-by-reference.

        This general implementation returns a NumPy copy of the entire data
        structure assigned to `_data`. This enforces that, unless a subclass
        specifically overrides this method, the data structure must be
        assigned to the private attribute `_data`.
        """
        try:
            with self._lock:
                return np.copy(self._data)
        except AttributeError:
            raise NotImplementedError()

    @property
    def unstructured(self):
        """Return structured data as regular `np.ndarray`."""
        raise NotImplementedError()

    @property
    def dtype(self):
        """Data type specification for the instance."""
        return self._dtype

    def initialize(self):
        """Initialize instance data; e.g. to zeros.

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
        with open(self.filename + '.json', 'w') as f:
            f.write(metadata_json)

    def _new_filename(self, data_dir='data', label=''):
        time_str = time.strftime("%y%m%d-%H%M%S", time.localtime())
        classname = type(self).__name__
        if label:
            label += '_'

        filename = '{}/{}_{}_{}{{}}'.format(data_dir, time_str,
                                            classname, label)
        # incremental counter to prevent overwrites
        # (based on existence of metadata file)
        count = 0
        while os.path.exists(filename.format(count) + '.json'):
            count += 1
        filename = filename.format(count)

        return filename

    def __deepcopy__(self, memo):
        # threading objects cannot be copied normally
        # & a new filename is needed
        mask = {'_lock': threading.Lock(),
                'filename': self._new_filename(self._data_dir, self._label)}
        return utils.deepcopy_mask(self, memo, mask)


class TimeSeries(Buffer):
    """Manages 2D time series data: rows of samples indexed in order of time.

    Data is stored in a NumPy structured array where `'time'` is the first
    field (named column) and the remaining fields correspond to the channel
    names passed during instantiation. Only the last `n_samples` are stored
    in memory this way, but all samples are written to disk when `record=True`.

    TODO:
        * Warning (error?) when timestamps are out of order
        * Marker channels
        * Per-channel units?
        * store_once behaviour is a bit awkward. What about long windows?
    """

    def __init__(self, ch_names, n_samples=2560, sfreq=None, record=True,
                 channel_fmt='f8', store_once=False, **kwargs):
        """Create a new `TimeSeries` object.

        Args:
            ch_names (List[str]): List of channel names.
            n_samples (int): Number of samples to store in memory.
            record (bool): Whether to record samples to disk.
            channel_fmt (str or type or List[str] or List[type]): Data type of
                channels. If a single string or type is passed, all channels
                will take that type. A list with the same length as `ch_names`
                may also be passed to independently specify channel types.
                Strings should conform to NumPy string datatype specifications;
                for example, a 64-bit float is specified as `'f8'`. Types may
                be Python base types (e.g. `float`) or NumPy base dtypes
                (e.g. `np.float64`).
            store_once (bool): Whether to stop storing data when window filled.
        """
        Buffer.__init__(self, **kwargs)

        self.sfreq = sfreq
        # if single dtype given, expand to number of channels
        try:
            np.dtype(channel_fmt)
            channel_fmt = [channel_fmt] * len(ch_names)
        except TypeError:
            pass

        try:
            self._dtype = np.dtype({'names': ["time"] + list(ch_names),
                                    'formats': [np.float64] + list(channel_fmt)
                                    })
        except ValueError:
            raise ValueError("Number of formats must match number of channels")

        self._record = record
        self._write = True
        self._store_once = store_once
        # write remaining data to file on program exit (e.g. quit())
        if record:
            atexit.register(self.write_to_file)

        self.initialize(int(n_samples))

    @classmethod
    def with_window(cls, ch_names, sfreq, window=10, **kwargs):
        """Constructs an instance based on a desired duration of storage.

        It is often convenient to specify the length of the array stored in
        memory as a duration. For example, a duration of 10 seconds might be
        specified so that the last 10 seconds of data will be available to an
        instance of `plot.Plotter`.

        This constructor also expects to be passed a nominal sampling frequency
        so that it can determine the number of samples corresponding to the
        desired duration. Note that duration is usually not evenly divisible by
        sampling frequency, and the number of samples stored will be rounded.

        Args:
            ch_names (List[str]): List of channel names.
            sfreq (int): Nominal sampling frequency of the input.
            window (float): Desired duration of live storage.
        """
        n_samples = int(window * sfreq)
        return cls(ch_names, n_samples, sfreq, **kwargs)

    def initialize(self, n_samples=None):
        """Initialize NumPy structured array for data storage.

        Args:
            n_samples (int): Number of samples (rows) in array.
        """
        if n_samples is None:
            n_samples = self.n_samples
        with self._lock:
            self._data = np.zeros((n_samples,), dtype=self._dtype)
        self._count = n_samples

    def update(self, timestamps, samples):
        """Append sample(s) to stored data.

        Args:
            timestamps (Iterable[np.float64]): Timestamp for each sample.
            samples (Iterable): Channel data.
                Data type(s) in `Iterable` correspond to the type(s) specified
                in `dtype`.

        TODO:
            * Sort timestamps/warn if unsorted?
        """
        new = self._format_samples(timestamps, samples)
        self.update_with_structured(new)

    def update_with_structured(self, new):
        """Append already structured data to stored data.

        Args:
            new (np.ndarray): Structured data (`dtype=self.dtype`).
        """
        self._new = new
        self._split_append(self._new)
        self.event_hook.fire()

    def write_to_file(self, force=False):
        """Write any unwritten samples to file.

        Args:
            force (bool): If `True`, forces writing of remaining samples
                regardless of the value of `record` passed at instantiation.
        """
        if self._record or force:
            with self._lock:
                with open(self.filename + ".csv", 'a') as f:
                    for row in self._data[max(0, self._count):]:
                        line = ','.join(str(n) for n in row)
                        f.write(line + '\n')
        self._count = self.n_samples

    def _split_append(self, new):
        # write out each time array contains only unwritten samples
        # however, last chunk added may push out some unwritten samples
        # therefore split appends before and after write_to_file
        cutoff = self._count
        if self._write:
            self._append(new[:cutoff])
            if self._count == 0:
                self.write_to_file()
                if self._store_once:
                    self._write = False
                else:
                    self._append(new[cutoff:])

    def _append(self, new):
        with self._lock:
            self._data = utils.push_rows(self._data, new)
        self._count -= len(new)

    def _format_samples(self, timestamps, samples):
        """Format data `numpy.ndarray` from timestamps and samples."""
        stacked = [(t,) + tuple(s) for t, s in zip(timestamps, samples)]
        try:
            stacked_ = np.array(stacked, dtype=self._dtype)
        except ValueError:
            raise ValueError(str(stacked))
        return stacked_

    def get_samples(self, last_n=0):
        """Return copy of channel data, without timestamps.

        Args:
            last_n (int): Number of most recent samples to return.
        """
        with self._lock:
            return np.copy(self._data[list(self.ch_names)][-last_n:])

    def get_unstructured(self, last_n=0):
        """Return unstructured copy of channel data, without timestamps.

        Args:
            last_n (int): Number of most recent samples to return.
        """
        samples = self.get_samples(last_n=last_n)
        try:
            return np.array(samples.tolist())
            #return samples.view((samples.dtype[0], self.n_chan))
        except ValueError as e:
            print(samples.shape, samples.dtype, self.n_chan)
            raise e
            raise ValueError("Cannot return unstructured data for " +
                             "channels with different datatypes/sample shapes")

    def get_timestamps(self, last_n=0):
        """Return copy of timestamps.

        Args:
            last_n (int): Number of most recent timestamps to return.
        """
        with self._lock:
            return np.copy(self._data['time'][-last_n:])

    @property
    def samples(self):
        """Copy of channel data, without timestamps."""
        return self.get_samples()

    @property
    def unstructured(self):
        """Unstructured copy of channel data, without timestamps."""
        return self.get_unstructured()

    @property
    def timestamps(self):
        """Copy of timestamps."""
        return self.get_timestamps()

    @property
    def last_samples(self):
        return np.copy(self._new)

    @property
    def last_sample(self):
        """Last-stored row (timestamp and sample)."""
        with self._lock:
            return np.copy(self._data[-1])

    @property
    def n_samples(self):
        """Number of samples stored in the NumPy array."""
        return self._data.shape[0]

    @property
    def n_new(self):
        """Number of samples received on last update."""
        return self._new.shape[0]

    @property
    def ch_names(self):
        """Channel names.

        Note:
            Does not include `'time'`.
        """
        return self.dtype.names[1:]

    @property
    def n_chan(self):
        """Number of channels."""
        return len(self.ch_names)


class Spectra(TimeSeries):
    """Manages a time series of spectral (e.g. frequency-domain) data.

    This is a constrained subclass of `TimeSeries`. Spectral data may be
    stored for multiple channels, but all channels will share the same
    spectral range (the `range` property).

    TODO:
        * What do timestamps mean here? Transformer-dependent?
    """

    def __init__(self, ch_names, indep_range, indep_name="Frequency",
                 values_dtype=None, **kwargs):
        """Create a new `Spectra` object.

        Args:
            ch_names (List[str]): List of channel names.
            indep_range (Iterable): Values of the independent variable.
            n_samples (int): Number of spectra updates to keep.
            indep_name (str): Name of the independent variable.
                Default: `"freq"`.
            values_dtype (type or np.dtype): Spectrum datatype.
                Default: `np.float64`.
        """
        if values_dtype is None:
            values_dtype = np.float64

        #try:
        #    if not sorted(indep_range) == list(indep_range):
        #        raise TypeError
        #except TypeError:
        #    raise TypeError("indep_range not a monotonic increasing sequence")

        super().__init__(ch_names=ch_names,
                         channel_fmt=(values_dtype, len(indep_range)),
                         **kwargs)

        self._range = indep_range
        self._indep_name = indep_name

    def update(self, timestamp, spectra):
        """Append a spectrum to stored data.

        Args:
            timestamp (np.float64): Timestamp for the current spectra.
            spectrum: Spectra for each of the channels.
                Should be a 2D iterable structure (e.g. list of lists, or
                `np.ndarray`) where the first dimension corresponds to channels
                and the second to the spectrum range.

        TODO:
            * May be able to remove this method if `TimeSeries` update method
              appends based on channel data type (see `TimeSeries` TODOs)
        """
        try:
            super(Spectra, self).update([timestamp], [spectra])
        except ValueError:
            msg = "cannot update with spectra of incorrect/inconsistent length"
            raise ValueError(msg)

    @property
    def range(self):
        return np.copy(self._range)

    @property
    def indep_name(self):
        return self._indep_name
