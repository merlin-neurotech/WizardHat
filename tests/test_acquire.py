""""""


import ble2lsl as b2l
import ble2lsl.devices as devices
from ble2lsl.devices import *
from wizardhat import acquire

import pkgutil
import time
import threading
from unittest import mock

import numpy as np
import pytest


# params: test dummies for all compatible devices; or test with no dummies
@pytest.fixture(scope='module',
                params=[(), devices.DEVICE_NAMES])
def dummy_streamers(request):
    """Instantiate a dummy streamer for each supported device."""
    dummies = []
    for device_name in request.param:
        dummy = b2l.Dummy(globals()[device_name])
        device = dummy._device
        subscriptions = {name for name in device.DEFAULT_SUBSCRIPTIONS
                         if device.PARAMS['streams']['nominal_srate'][name] > 0}
        source_id = "{}-{}".format(device.NAME, 'DUMMY')
        dummies.append((dummy, device, source_id, subscriptions))

    def teardown():
        for dummy, _, _, _ in dummies:
            dummy.stop()
            del(dummy)
    request.addfinalizer(teardown)

    return dummies


@pytest.fixture(scope='module')
def streams_dict(dummy_streamers):
    streams_dict = acquire.get_lsl_streams()
    return streams_dict


def test_get_lsl_streams(streams_dict, dummy_streamers):
    # detected source_ids
    source_ids = streams_dict.keys()

    # if no source_ids detected, this should be the no-dummies case
    if list(source_ids) == []:
        assert dummy_streamers == []

    for dummy, device, source_id, subscriptions in dummy_streamers:
        # Dummy detected for each device?
        assert source_id in source_ids
        # all default subscriptions streamed?
        assert (set(streams_dict[source_id].keys())
                == set(subscriptions))

        for stream_type, stream in streams_dict[source_id].items():
            # stream corresponds to device?
            assert stream.source_id() == source_id
            # stream follows naming convention?
            stream_name = "{}-{}".format(device.NAME, stream_type)
            assert stream.name() == stream_name
            # other metadata is correct?
            assert (stream.type()
                    == device.PARAMS['streams']['type'][stream_type])
            nominal_srate = (device
                             .PARAMS['streams']['nominal_srate'][stream_type])
            assert (stream.nominal_srate() == nominal_srate)
            assert (stream.channel_count()
                    == device.PARAMS['streams']['channel_count'][stream_type])


@pytest.fixture(scope='module')
def stream_inlets(streams_dict, dummy_streamers):
    # NOTE: the arrangement of stream_inlets may seem redundant;
    # when functioning normally `stream_inlets[source_id]` should have one
    # key (equal to `source_id`); i.e. the same key will be nested
    # BUT this will not necessarily be true if something is wrong
    stream_inlets = {}
    for dummy, device, source_id, subscriptions in dummy_streamers:
        inlets = acquire.get_lsl_inlets(streams_dict,
                                        with_source_ids=(source_id,))
        stream_inlets[source_id] = inlets
    return stream_inlets


def test_get_lsl_inlets(streams_dict, dummy_streamers, stream_inlets):
    """Do the acquired inlets contain the expected metadata, etc.?"""
    if dummy_streamers == []:
        # no-dummies case
        assert acquire.get_lsl_inlets(streams_dict) == {}

    for dummy, device, source_id, subscriptions in dummy_streamers:
        inlets = stream_inlets[source_id]
        # only contains key for selected source id?
        assert set(inlets.keys()) == {source_id}
        # inlet keys correspond to default subscriptions?
        assert (set(inlets[source_id].keys())
                == set(subscriptions))

        for stream_type, inlet in inlets[source_id].items():
            info = inlet.info()
            # StreamInlet corresponds to same outlet as StreamInfo?
            assert info.uid() == streams_dict[source_id][stream_type].uid()
            # TODO: test that more of the metadata is the same?
            # or assume the rest must be correct based on pylsl behaviour
            # when instantiating a StreamInlet using a StreamInfo?

        # check that with_type works for single type selection
        for stream_type in subscriptions:
            inlets = acquire.get_lsl_inlets(streams_dict,
                                            with_source_ids=(source_id,),
                                            with_types=(stream_type,))
            assert set(inlets.keys()) == {source_id}
            assert (set(inlets[source_id].keys())) == {stream_type}

        # TODO: check that with_type works for more arbitrary type selections


def test_get_ch_names(dummy_streamers, stream_inlets):
    """Does `acquire.ch_names` reflect the channel names in the device file?"""
    for _, device, source_id, subscriptions in dummy_streamers:
        inlets = stream_inlets[source_id]
        for stream_type in subscriptions:
            ch_names = acquire.get_ch_names(inlets[source_id][stream_type]
                                            .info())
            assert (device.PARAMS['streams']['ch_names'][stream_type]
                    == tuple(ch_names))


def test_dejitter_timestamps():
    """This is basically a reference implementation of `dejitter_timestamps`
    against which it is compared..."""
    n_steps = 100
    n_tests = 50
    sfreqs = np.linspace(1, 5000, n_tests).astype(int)
    last_times = np.random.randint(-100, 100, size=n_tests)
    test_timestamps = np.random.random((n_tests, n_steps)) + np.arange(n_steps)
    expected_timestamps = [np.arange(n_steps)/sfreq + last_times[i] + 1/sfreq
                           for i, sfreq in enumerate(sfreqs)]
    for i, args in enumerate(zip(test_timestamps, sfreqs, last_times)):
        dejittered = acquire.dejitter_timestamps(*args)
        # there may be some floating-point errors, so just make sure the
        # difference is tiny
        assert np.all((dejittered - expected_timestamps[i]) < 1e-14)


def construct_receiver_no_id(source_id=None, **kwargs):
    """Used to force manual selection among multiple sources."""
    return acquire.Receiver(source_id=None, **kwargs)


@pytest.fixture(scope='class', params=[acquire.Receiver,
                                       construct_receiver_no_id])
def dummy_receivers(request, dummy_streamers):
    """Provides `acquire.Receiver` objects for dummy devices.

    Either constructs by giving source ID, or by mocking user input.
    """
    receivers = {}
    for idx, (_, _, source_id, _) in enumerate(dummy_streamers):
        with mock.patch('builtins.input', side_effect=str(idx)):
            receiver = request.param(source_id=source_id, autostart=False)
        receivers[source_id] = receiver

    def teardown():
        for sid, receiver in receivers.items():
            receiver.stop()
            del(receiver)
    request.addfinalizer(teardown)

    return receivers


@pytest.mark.usefixtures("dummy_streamers", "dummy_receivers")
class TestReceiver:
    """Tests the `acquire.Receiver` objects provided by `dummy_receivers`."""

    def test_multiple_streams(self, dummy_streamers, dummy_receivers):
        """Make sure a receiver was made for each available source."""
        dummy_ids = [source_id for _, _, source_id, _ in dummy_streamers]
        source_ids = [receiver._source_id
                      for _, receiver in dummy_receivers.items()]
        assert set(source_ids) == set(dummy_ids)

    def test_metadata(self, dummy_streamers, dummy_receivers):
        """Test whether the `Receiver` object contains expected metadata."""
        for dummy, device, source_id, subscriptions in dummy_streamers:
            receiver = dummy_receivers[source_id]
            stream_params = device.PARAMS['streams']

            # check metadata
            for stream_type in subscriptions:
                assert (receiver.sfreq[stream_type]
                        == stream_params['nominal_srate'][stream_type])
                assert (receiver.n_chan[stream_type]
                        == stream_params['channel_count'][stream_type])
                assert (receiver.ch_names[stream_type]
                        == list(stream_params['ch_names'][stream_type]))

    def test_streaming(self, dummy_streamers, dummy_receivers):
        """Test whether streaming threads and data transmission work."""
        for dummy, device, source_id, subscriptions in dummy_streamers:
            receiver = dummy_receivers[source_id]
            # basic thread behaviour (start on `receiver.start()`)
            for thread in receiver._threads.values():
                assert not thread.is_alive()
            receiver.start()
            for thread in receiver._threads.values():
                assert thread.is_alive()

            # TODO: compare data (use pre-defined data)

            # NOTE: some threads may take a while to stop,
            #       not sure how to assert this properly
            receiver.stop()
            #for thread in receiver._threads.values():
            #    assert not thread.is_alive()
