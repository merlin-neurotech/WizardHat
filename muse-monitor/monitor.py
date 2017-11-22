#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""

import classify
import online
import utils

import numpy as np


if __name__ == '__main__':
    streamer = online.LSLStreamer()
    recorder = online.LSLRecorder(streamer)
    sfreq = streamer.sfreq

    # initial messages
    welcome = """\nEEG data will be recorded. During each trial, please be as
    still as possible while following each set of instructions.\n"""
    print(welcome)

    # record trials
    specs = [{'length': 30, 'label': 0,
              'msg': "Please focus visually and concentrate."},
             {'length': 30, 'label': 1,
              'msg': "Please close your eyes and relax."}]
    streamer.start()
    trials = recorder.record_trials(specs)
    epoched = {label: utils.epoching(data['samples'], 100, samples_overlap=0)
               for label, data in trials.items()}
    feature_matrices = {label: utils.calc_feature_matrix(trial[0], sfreq)
                        for label, trial in epoched.items()}

    classifier, mu_ft, std_ft = classify.train_binary_svm(feature_matrices)

    print("Press Enter to begin online classification.")
    input()
    streamer.updated.clear()
    while True:
        streamer.updated.wait()
        with streamer.lock:
            samples = np.copy(streamer.data['samples'])
            streamer.updated.clear()
        epochs, remainder = utils.epoching(samples, streamer.n_samples)
        feature_matrix = utils.calc_feature_matrix(epochs, sfreq)
        x = (feature_matrix - mu_ft) / std_ft
        y_hat = classifier.predict(x)
        print(y_hat)
