"""Interfacing parameters for the Muse headband (2016 version)."""


PARAMS = dict(
    manufacturer='Muse',
    units='microvolts',
    ch_names=('TP9', 'AF7', 'AF8', 'TP10', 'Right AUX'),
    ch_uuids=(
        '273e0003-4c4d-454d-96be-f03bac821358',
        '273e0004-4c4d-454d-96be-f03bac821358',
        '273e0005-4c4d-454d-96be-f03bac821358',
        '273e0006-4c4d-454d-96be-f03bac821358',
        '273e0007-4c4d-454d-96be-f03bac821358',
    ),
    packet_dtypes=dict(index='uint:16', ch_value='uint:12'),
    ble=dict(
        handle=0x000e,
        stream_on=(0x02, 0x64, 0x0a),
        stream_off=(0x02, 0x68, 0x0a),
    ),
)
"""General Muse headset parameters."""


STREAM_PARAMS = dict(
    name='Muse',
    type='EEG',
    channel_count=5,
    nominal_srate=256,
    channel_format='float32',
)
"""Muse headset parameters for constructing `pylsl.StreamInfo`."""
