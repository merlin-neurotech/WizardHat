import pygatt
from pygatt.backends import BLEAddressType
import binascii

name = 'Ganglion'
adapter = pygatt.BGAPIBackend()
adapter.start()
devices = adapter.scan()

ble = dict(service='fe84',
recieve='2d30c082f39f4ce6923f3484ea480596',
send="2d30c083f39f4ce6923f3484ea480596",
disconnect="2d30c084f39f4ce6923f3484ea480596")

for device in devices:
    if name in device['name']:
        address = device['address']

device = adapter.connect(address, address_type=BLEAddressType.random)

device.char_write_handle(ble['service'], value=ble['recieve'],wait_for_response=False)
print('Streaming')
