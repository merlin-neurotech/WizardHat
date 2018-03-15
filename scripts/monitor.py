
"""

"""

from wizardhat import acquire
import ble2lsl

if __name__ == '__main__':
    outlet_streamer = ble2lsl.LSLOutletStreamer()
    # outlet_streamer = ble2lsl.LSLOutletDummy()
    streamer = acquire.LSLStreamer()
