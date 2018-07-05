"""Utilities for use within BLE2LSL."""


def invert_map(dict_):
    """Invert the keys and values in a dict."""
    inverted = {v: k for k, v in dict_.items()}
    return inverted


def bad_data_size(data, size, data_type="packet"):
    if len(data) != size:
        print('Wrong size for {}, {} instead of {} bytes'
              .format(data_type, len(data), size))
        return True
    return False
