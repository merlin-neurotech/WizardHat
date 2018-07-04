"""Utilities for use within BLE2LSL."""

def invert_map(dict_):
    """Invert the keys and values in a dict."""
    inverted = {v: k for k, v in dict_.items()}
    return inverted
