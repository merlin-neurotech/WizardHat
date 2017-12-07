#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Binary SVM classifier of concentrating vs. relaxating states from EEG data."""

from musemonitor import classify, online, utils


channels = ['TP9', 'AF7', 'AF8', 'TP10']


# trial parameters
lengths = [5, 5]  # seconds
labels = [0, 1]
messages = ["Please focus visually and concentrate.",
            "Please close your eyes and relax."]

# calculation parameters
epoch_size = 100

if __name__ == '__main__':
    streamer = online.LSLStreamer()
    recorder = online.LSLRecorder(streamer)
    sfreq = streamer.sfreq

    # initial messages
    welcome = """
    EEG data will be recorded. During each trial, please be as still as \
    possible while following each set of instructions.
    """
    print(welcome)

    # record trials
    trials = recorder.record_trials(lengths, messages)

    # calculate epochs and features (EEG band means) for each trial
    epoched = [utils.epoching(data['channels'], epoch_size) for data in trials]
    feature_matrices = [utils.calc_feature_matrix(trial[0], sfreq)
                        for trial in epoched]

    classifier, mu_ft, std_ft = classify.train_binary_svm(feature_matrices,
                                                          labels)

    print("Press Enter to begin online classification.")
    input()
    streamer.proceed = False
    streamer.join()

    streamer2 = online.LSLStreamer(inlet=streamer.inlet, window=1)

    streamer2.updated.clear()
    while True:
        streamer2.updated.wait()
        data = streamer2.get_data()
        streamer2.updated.clear()
        epochs, remainder = utils.epoching(data['channels'],
                                           streamer2.n_samples)
        feature_matrix = utils.calc_feature_matrix(epochs, sfreq)
        x = (feature_matrix - mu_ft) / std_ft
        y_hat = classifier.predict(x)
        print(y_hat)
