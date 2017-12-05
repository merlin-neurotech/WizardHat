# -*- coding: utf-8 -*-
"""
"""

import numpy as np
from sklearn import svm


def train_binary_svm(feature_matrices):
    if len(feature_matrices) != 2:
        e_msg = "Binary classification requires exactly two feature matrices."
        raise(ValueError(e_msg))
    try:
        classes = [np.full((feature_matrix.shape[0], 1), label)
                   for label, feature_matrix in zip(labels, feature_matrices)]
    except AttributeError:
        e_msg = """Feature matrices must be passed as values
               of a `dict` with class labels as keys."""
        raise(ValueError(e_msg))
    y = np.ravel(np.concatenate([*classes], axis=0))
    features_all = np.concatenate([*feature_matrices], axis=0)

    mu_ft = np.mean(features_all, axis=0)
    std_ft = np.std(features_all, axis=0)

    X = (features_all - mu_ft) / std_ft

    classifier = svm.SVC()
    classifier.fit(X, y)

    return classifier, mu_ft, std_ft
