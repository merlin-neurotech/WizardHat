"""Utilities for use within the WizardHat API and its extensions.

The intention of this module is to avoid unnecessary repetition of programming
constructs. This includes routine manipulations of strings or data structures
(but not calculations/transformations on data itself), interfacing with the
operating system, or general-purpose method decorators.
"""

import copy
import os

import numpy as np


def deepcopy_mask(obj, memo, mask=None):
    """Generalized method for deep copies of objects.

    Certain types of attributes cannot be copied naively by `copy.deepcopy`;
    for example, `threading.Lock` objects. These may be manually specified in
    the `mask` argument.

    Args:
        obj (object): The object to be copied.
        mask (Dict[str, object]): Attributes to be replaced manually.
            Keys are attribute names and values are new attribute values.
        memo (dict): Tracks already copied objects to prevent a recursive loop.
            See the documentation for the standard module `copy`.
    """
    if mask is None:
        mask = {}
    cls = obj.__class__
    # copy object
    clone = cls.__new__(cls)
    memo[id(obj)] = clone
    # copy object attributes
    for k, v in obj.__dict__.items():
        if k in mask:
            setattr(clone, k, mask[k])
        else:
            setattr(clone, k, copy.deepcopy(v, memo))
    return clone


def push_rows(arr, rows):
    """Append rows to an array and discard the same number from the front.

    The arguments may be any sequence (`Iterable`) that NumPy can convert to
    `np.ndarray`, but both must have the same number of columns.

    Args:
        arr (Iterable): Array onto which to append the rows.
        rows (Iterable): Rows to be appended.
    """
    n = arr.shape[0]
    arr = np.concatenate([arr, rows], axis=0)
    return arr[-n:]


def makedirs(filepath):
    """Create a directory tree if it does not exist yet, based on a filepath.

    Args:
        filepath (str): The path for which to create directories.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
