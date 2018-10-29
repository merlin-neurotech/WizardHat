""""""


import ble2lsl as b2l
import ble2lsl.devices as devices

from wizardhat import acquire

import pkgutil

import pytest
import pytest_mock as mock

# params: test dummies for all compatible devices; or test with no dummies
@pytest.fixture(scope='module',
                params=[devices.DEVICE_NAMES, ()])
def dummy_streamers(request):
    """Instantiate a dummy streamer for each supported device."""
    from ble2lsl.devices import *
    dummies = []
    for device_name in request.param:
        dummy = b2l.Dummy(globals()[device_name])
        device = dummy._device
        source_id = "{}-{}".format(device.name, 'DUMMY')
        dummies.append((dummy, device, source_id))

    def teardown():
        for dummy in dummies:
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

    for dummy, device, source_id in dummy_streamers:
        # Dummy detected for each device?
        assert source_id in source_ids
        # all default subscriptions streamed?
        assert (set(streams_dict[source_id].keys())
                == set(device.DEFAULT_SUBSCRIPTIONS)

        for stream_type, stream in streams_dict[source_id]:
            # stream corresponds to device?
            assert stream.source_id() == source_id
            # stream follows naming convention?
            stream_name = "{}-{}".format(device.name, stream_type)
            assert stream.name() == stream_name
            # other metadata is correct?
            assert (stream.type()
                    == device.PARAMS['streams']['type'][stream_type])
            nominal_srate = (device
                             .PARAMS['streams']['nominal_srate'][streamtype])
            assert (stream.nominal_srate() == nominal_srate)
            assert (stream.channel_count()
                    == device.PARAMS['streams']['channel_count'][streamtype])

def test_get_lsl_inlets(streams_dict, dummy_streamers):
    if dummy_streamers == []:
        # no-dummies case
        assert acquire.get_lsl_inlets(streams_dict) == {}

    for dummy, device, source_id in dummy_streamers:
        inlets = acquire.get_lsl_inlets(streams_dict,
                                        with_source_ids=(source_id,))
        # only contains key for selected source id?
        assert set(inlets.keys()) == {source_id}
        # inlet keys correspond to default subscriptions?
        assert (set(inlets[source_id].keys())
                == set(device.DEFAULT_SUBSCRIPTIONS))

        for stream_type, inlet in inlets[source_id]:
            info = inlet.info()
            # StreamInlet corresponds to same outlet as StreamInfo?
            assert info.uid() == streams_dict[source_id][stream_type].uid()
            # TODO: test that more of the metadata is the same?
            # or assume the rest must be correct based on pylsl behaviour
            # when instantiating a StreamInlet using a StreamInfo?

        # check that with_type works for single type selection
        for stream_type in device.DEFAULT_SUBSCRIPTIONS:
            inlets = acquire.get_lsl_inlets(streams_dict,
                                            with_source_ids=(source_id,),
                                            with_types=(stream_type,))
            assert set(inlets.keys()) == {source_id}
            assert (set(inleys[source_id].keys())) == {stream_type}

        # TODO: check that with_type works for more arbitrary type selections

@pytest.fixture(scope='class')
def dummy_receivers(dummy_streamers):
    receivers = {source_id: acquire.Receiver(source_id=source_id)
                 for _, _, source_id in dummy_streamers}
    return receivers


class TestReceiver:

    def test_multiple_streams(self, dummy_streamers):
        if len(dummy_streamers) > 1:
            pass

    def test_metadata(self, dummy_streamers, dummy_receivers):
        for dummy, device, source_id in dummy_streamers:
            receiver = dummy_receivers[source_id]
            stream_params = device.PARAMS['streams']
            # check metadata
            for stream_type in device.DEFAULT_SUBSCRIPTIONS:
                assert (receiver.sfreq[stream_type]
                        == stream_params['nominal_srate'][stream_type])
                assert (receiver.n_chan[stream_type]
                        == stream_params['channel_count'][stream_type])
                assert (receiver.ch_names[stream_type]
                        == stream_params['ch_names'][stream_type])

    def test_streaming(self, dummy_streamers):
        for dummy, device, source_id in dummy_streamers:
            receiver = dummy_receivers[source_id]
            # basic thread behaviour (start on `receiver.start()`)
            for thread in receiver._threads:
                assert not thread.is_alive()
            receiver.start()
            for thread in receiver._threads:
                assert thread.is_alive()

            # TODO: compare data (use pre-defined data)
            
            receiver.stop()
            for thread in receiver._threads:
                assert not thread.is_alive()
