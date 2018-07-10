"""Utilities for use within BLE2LSL."""

from warnings import warn

def invert_map(dict_):
    """Invert the keys and values in a dict."""
    inverted = {v: k for k, v in dict_.items()}
    return inverted


def bad_data_size(data, size, data_type="packet"):
    """Return `True` if length of `data` is not `size`."""
    if len(data) != size:
        warn('Wrong size for {}, {} instead of {} bytes'
             .format(data_type, len(data), size))
        return True
    return False


def dict_partial_from_keys(keys):
    """Return a function that constructs a dict with predetermined keys."""
    def dict_partial(values):
        return dict(zip(keys, values))
    return dict_partial
