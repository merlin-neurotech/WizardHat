"""Interfacing parameters for the OpenBCI Ganglion Board."""

PARAMS = dict(
    manufacturer='OpenBCI',
    units='microvolts',
    ch_names=('1','2','3','4'),
    ble=dict(
    service='fe84',
    receive='2d30c082f39f4ce6923f3484ea480596',
    send="2d30c083f39f4ce6923f3484ea480596",
    disconnect="2d30c084f39f4ce6923f3484ea480596",),
    packet_dtypes=dict(index='uint:16', ch_value='uint:12')
    )
STREAM_PARAMS = dict(
    name='Ganglion',
    type='EEG',
    channel_count=4,
    nominal_srate=200,
    channel_format='float32' 
)
