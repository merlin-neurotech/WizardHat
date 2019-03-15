"""Utilities for use within the WizardHat API and its extensions.

The intention of this module is to avoid unnecessary repetition of programming
constructs. This includes routine manipulations of strings or data structures
(but not calculations/transformations on data itself), interfacing with the
operating system, or general-purpose method decorators.
"""

import copy
import os

import numpy as np


class EventHook:
    """Handler for multiple callbacks triggered by a single event.

    Callbacks may be registered with an `EventHook` instance using the
    incremental add operator (`event_hook_instance += some_callback_function`),
    and deregistered by incremental subtraction. When the instance's `fire`
    method is called (i.e. upon some event), all of the registered callback
    functions will also be called.

    The primary use for this class is in `Buffer` classes, whose `EventHook`
    instances allow them to call the update functions of all downstream objects
    (e.g. `Plotter` or `Transformer` instances).

    TODO:
        * multiprocessing: spread the workload over several processes; maybe
        give the option to use either threading or multiprocessing for a given
        callback, depending on its complexity (IO vs. calculations)
    """
    def __init__(self):
        self._handlers = []

    def __iadd__(self, handler):
        self._handlers.append(handler)
        return self

    def __isub__(self, handler):
        self._handlers.remove(handler)
        return self

    def fire(self, *args, **keywargs):
        """Call all registered callback functions."""
        for handler in self._handlers:
            handler(*args, **keywargs)

    def clear_handlers(self, in_object):
        """Deregister all methods of a given object."""
        for handler in self.__handlers:
            if handler.__self__ == in_object:
                self -= handler


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


class always_greater(int):
    """Instances always compare `True` when on the left hand side of `>`."""
    def __gt__(self, other):
        return True


def menu_numlist(options, menu_intro='', max_tries=always_greater(),
                 raise_on_failure=False):
    """Print a numbered list of options and return the first valid choice.

    Args:
        menu_intro (str): String to print before displaying the menu.
        max_tries (int): Maximum number of times to retry on invalid response.
        raise_on_failure (bool): If `True`, raise exception on no more tries.
    """
    print('\n', menu_intro)
    print("Choose one of following:")
    menu = '\n'.join("{}. {}".format(i, item)
                     for i, item in enumerate(options))
    print(menu, '\n')

    select = "Selection [0-{}]: ".format(len(options) - 1)
    response = None
    count = 0
    while response is None and max_tries > count:
        try:
            count += 1
            response = options[int(input(select))]
        except (ValueError, IndexError):
            print("Invalid selection! Try again.")

    if response is None and raise_on_failure:
        raise ValueError("No menu option selected in given number of tries")

    return response



def makedirs(filepath):
    """Create a directory tree if it does not exist yet, based on a filepath.

    Args:
        filepath (str): The path for which to create directories.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)


def next_pow2(n):
    """Return the nearest power of 2 greater than a number."""
    return int(2 ** np.ceil(np.log2(n)))
