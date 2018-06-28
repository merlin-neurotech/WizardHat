"""Interfacing parameters for the OpenBCI Ganglion Board."""
from pygatt import BLEAddressType
import struct
PARAMS = dict(
    manufacturer='OpenBCI',
    units='microvolts',
    ch_names=('A', 'B', 'C', 'D'),
    chunk_size=1,
    ble=dict(
        address_type=BLEAddressType.random,
        service='fe84',
        # Recieve UUID TODO Naming scheme for the ble2lsl subscription for all devices
        uuid=['2d30c082f39f4ce6923f3484ea480596'],
        send="2d30c083f39f4ce6923f3484ea480596",
        stream_on=b'b',
        stream_off=b's',
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
import numpy as np
scale_fac_uVolts_per_count = 1200 / (8388607.0 * 1.5 * 51.0)
scale_fac_accel_G_per_count = 0.000016

class PacketManager():
  """ Called by bluepy (handling BLE connection) when new data arrive, parses samples. """
  def __init__(self, scaling_output = True):
      # holds samples until OpenBCIBoard claims them
      # detect gaps between packets
      self.last_id = -1
      self.packets_dropped = 0
      # save uncompressed data to compute deltas
      self.lastChannelData = [0, 0, 0, 0]
      # 18bit data got here and then accelerometer with it
      self.lastAcceleromoter = [0, 0, 0]
      # when the board is manually set in the right mode (z to start, Z to stop), impedance will be measured. 4 channels + ref
      self.lastImpedance = [0, 0, 0, 0, 0]
      self.scaling_output = scaling_output
      # handling incoming ASCII messages
      self.receiving_ASCII = False


  def handleNotification(self, cHandle, data):
    if len(data) < 1:
      print('Warning: a packet should at least hold one byte...')
      return
    self.parse(data)

  """
    PARSER:
    Parses incoming data packet into OpenBCISample -- see docs. Will call the corresponding parse* function depending on the format of the packet.
  """
  def parse(self, packet): 
    unpac = packet
    start_byte = unpac[0]
    # Give the informative part of the packet to proper handler -- split between ID and data bytes
    # Raw uncompressed
    if start_byte == 0:
      self.receiving_ASCII = False
      self.parseRaw(start_byte, unpac[1:])
    # 18-bit compression with Accelerometer
    elif start_byte >= 1 and start_byte <= 100:
      self.receiving_ASCII = False
      self.parse18bit(start_byte, unpac[1:])
    # 19-bit compression without Accelerometer
    elif start_byte >=101 and start_byte <= 200:
      self.receiving_ASCII = False
      self.parse19bit(start_byte-100, unpac[1:])
    # Impedance Channel
    elif start_byte >= 201 and start_byte <= 205:
      self.receiving_ASCII = False
      self.parseImpedance(start_byte, packet[1:])
    # Part of ASCII -- TODO: better formatting of incoming ASCII
    elif start_byte == 206:
      print("%\t" + str(packet[1:]))
      self.receiving_ASCII = True
      #self.time_last_ASCII = timeit.default_timer() 
      
    # End of ASCII message
    elif start_byte == 207:
      print("%\t" + str(packet[1:]))
      print ("$$$")
      self.receiving_ASCII = False
    else:
      print("Warning: unknown type of packet: " + str(start_byte))

  def parseRaw(self, packet_id, packet):
    """ Dealing with "Raw uncompressed" """
    if len(packet) != 19:
      print('Wrong size, for raw data' + str(len(data)) + ' instead of 19 bytes')
      return

    chan_data = []
    # 4 channels of 24bits, take values one by one
    for i in range(0,12,3):
      chan_data.append(conv24bitsToInt(packet[i:i+3]))
    # save uncompressed raw channel for future use and append whole sample
    self.pushSample(packet_id, chan_data, self.lastAcceleromoter, self.lastImpedance)
    self.lastChannelData = chan_data
    self.updatePacketsCount(packet_id)

  def parse19bit(self, packet_id, packet):
    """ Dealing with "19-bit compression without Accelerometer" """
    if len(packet) != 19:
      print('Wrong size, for 19-bit compression data' + str(len(data)) + ' instead of 19 bytes')
      return

    # should get 2 by 4 arrays of uncompressed data
    deltas = decompressDeltas19Bit(packet)
    # the sample_id will be shifted
    delta_id = 1
    for delta in deltas:
      # convert from packet to sample id
      sample_id = (packet_id - 1) * 2 + delta_id
      # 19bit packets hold deltas between two samples
      # TODO: use more broadly numpy
      full_data = list(np.array(self.lastChannelData) - np.array(delta))
      # NB: aux data updated only in 18bit mode, send values here only to be consistent
      self.pushSample(sample_id, full_data, self.lastAcceleromoter, self.lastImpedance)
      self.lastChannelData = full_data
      delta_id += 1
    self.updatePacketsCount(packet_id)


  def parse18bit(self, packet_id, packet):
    """ Dealing with "18-bit compression without Accelerometer" """
    if len(packet) != 19:
      print('Wrong size, for 18-bit compression data' + str(len(data)) + ' instead of 19 bytes')
      return

    # accelerometer X
    if packet_id % 10 == 1:
      self.lastAcceleromoter[0] = conv8bitToInt8(packet[18])
    # accelerometer Y
    elif packet_id % 10 == 2:
      self.lastAcceleromoter[1] = conv8bitToInt8(packet[18])
    # accelerometer Z
    elif packet_id % 10 == 3:
      self.lastAcceleromoter[2] = conv8bitToInt8(packet[18])
        
    # deltas: should get 2 by 4 arrays of uncompressed data
    deltas = decompressDeltas18Bit(packet[:-1])
    # the sample_id will be shifted
    delta_id = 1
    for delta in deltas:
      # convert from packet to sample id
      sample_id = (packet_id - 1) * 2 + delta_id
      # 19bit packets hold deltas between two samples
      # TODO: use more broadly numpy
      full_data = list(np.array(self.lastChannelData) - np.array(delta))
      self.pushSample(sample_id, full_data, self.lastAcceleromoter, self.lastImpedance)
      self.lastChannelData = full_data
      delta_id += 1
    self.updatePacketsCount(packet_id)


  def parseImpedance(self, packet_id, packet):
    """ Dealing with impedance data. packet: ASCII data. NB: will take few packet (seconds) to fill"""
    if packet[-2:] != "Z\n":
      print('Wrong format for impedance check, should be ASCII ending with "Z\\n"')

    # convert from ASCII to actual value
    imp_value = int(packet[:-2])
    # from 201 to 205 codes to the right array size
    self.lastImpedance[packet_id- 201] =  imp_value
    self.pushSample(packet_id - 200, self.lastChannelData, self.lastAcceleromoter, self.lastImpedance)

    
  def pushSample(self, sample_id, chan_data, aux_data, imp_data):
    """ Add a sample to inner stack, setting ID and dealing with scaling if necessary. """
    if self.scaling_output:
      chan_data = np.array([chan_data]) * scale_fac_uVolts_per_count
      #aux_data = np.array(aux_data) * scale_fac_accel_G_per_count
    #sample = OpenBCISample(sample_id, chan_data, aux_data, imp_data)
      
    self.sample = [sample_id, chan_data.T]
    
  def updatePacketsCount(self, packet_id):
    """Update last packet ID and dropped packets"""
    if self.last_id == -1:
      self.last_id = packet_id
      self.packets_dropped  = 0
      return
    # ID loops every 101 packets
    if packet_id > self.last_id:
      self.packets_dropped = packet_id - self.last_id - 1
    else:
      self.packets_dropped = packet_id + 101 - self.last_id - 1
    self.last_id = packet_id
    if self.packets_dropped > 0:
        pass
        #print("Warning: dropped " + str(self.packets_dropped) + " packets.")

  def getSamples(self):
    """ Retrieve and remove from buffer last samples. """
    unstack_samples = self.samples
    self.samples = []
    return unstack_samples

  def getMaxPacketsDropped(self):
    """ While processing last samples, how many packets were dropped?"""
    # TODO: return max value of the last samples array?
    return self.packets_dropped



"""
  DATA conversion, for the most part courtesy of OpenBCI_NodeJS_Ganglion

"""
  
def conv24bitsToInt(unpacked):
  """ Convert 24bit data coded on 3 bytes to a proper integer """ 
  if len(unpacked) != 3:
    raise ValueError("Input should be 3 bytes long.")

  # FIXME: quick'n dirty, unpack wants strings later on
  literal_read = struct.pack('3B', unpacked[0], unpacked[1], unpacked[2])

  #3byte int in 2s compliment
  if (unpacked[0] > 127):
    pre_fix = bytes(bytearray.fromhex('FF')) 
  else:
    pre_fix = bytes(bytearray.fromhex('00'))

  literal_read = pre_fix + literal_read;

  #unpack little endian(>) signed integer(i) (makes unpacking platform independent)
  myInt = struct.unpack('>i', literal_read)[0]

  return myInt

def conv19bitToInt32(threeByteBuffer):
  """ Convert 19bit data coded on 3 bytes to a proper integer (LSB bit 1 used as sign). """ 
  if len(threeByteBuffer) != 3:
    raise ValueError("Input should be 3 bytes long.")

  prefix = 0;

  # if LSB is 1, negative number, some hasty unsigned to signed conversion to do
  if threeByteBuffer[2] & 0x01 > 0:
    prefix = 0b1111111111111;
    return ((prefix << 19) | (threeByteBuffer[0] << 16) | (threeByteBuffer[1] << 8) | threeByteBuffer[2]) | ~0xFFFFFFFF
  else:
    return (prefix << 19) | (threeByteBuffer[0] << 16) | (threeByteBuffer[1] << 8) | threeByteBuffer[2]

def conv18bitToInt32(threeByteBuffer):
  """ Convert 18bit data coded on 3 bytes to a proper integer (LSB bit 1 used as sign) """ 
  if len(threeByteBuffer) != 3:
    raise Valuerror("Input should be 3 bytes long.")

  prefix = 0;

  # if LSB is 1, negative number, some hasty unsigned to signed conversion to do
  if threeByteBuffer[2] & 0x01 > 0:
    prefix = 0b11111111111111;
    return ((prefix << 18) | (threeByteBuffer[0] << 16) | (threeByteBuffer[1] << 8) | threeByteBuffer[2]) | ~0xFFFFFFFF
  else:
    return (prefix << 18) | (threeByteBuffer[0] << 16) | (threeByteBuffer[1] << 8) | threeByteBuffer[2]
  
def conv8bitToInt8(byte):
  """ Convert one byte to signed value """ 

  if byte > 127:
    return (256-byte) * (-1)
  else:
    return byte
  
def decompressDeltas19Bit(buffer):
  """
  Called to when a compressed packet is received.
  buffer: Just the data portion of the sample. So 19 bytes.
  return {Array} - An array of deltas of shape 2x4 (2 samples per packet and 4 channels per sample.)
  """ 
  if len(buffer) != 19:
    raise ValueError("Input should be 19 bytes long.")
  
  receivedDeltas = [[0, 0, 0, 0],[0, 0, 0, 0]]

  # Sample 1 - Channel 1
  miniBuf = [
      (buffer[0] >> 5),
      ((buffer[0] & 0x1F) << 3 & 0xFF) | (buffer[1] >> 5),
      ((buffer[1] & 0x1F) << 3 & 0xFF) | (buffer[2] >> 5)
    ]

  receivedDeltas[0][0] = conv19bitToInt32(miniBuf)

  # Sample 1 - Channel 2
  miniBuf = [
      (buffer[2] & 0x1F) >> 2,
      (buffer[2] << 6 & 0xFF) | (buffer[3] >> 2),
      (buffer[3] << 6 & 0xFF) | (buffer[4] >> 2)
    ]
  receivedDeltas[0][1] = conv19bitToInt32(miniBuf)

  # Sample 1 - Channel 3
  miniBuf = [
      ((buffer[4] & 0x03) << 1 & 0xFF) | (buffer[5] >> 7),
      ((buffer[5] & 0x7F) << 1 & 0xFF) | (buffer[6] >> 7),
      ((buffer[6] & 0x7F) << 1 & 0xFF) | (buffer[7] >> 7)
    ]
  receivedDeltas[0][2] = conv19bitToInt32(miniBuf)

  # Sample 1 - Channel 4
  miniBuf = [
      ((buffer[7] & 0x7F) >> 4),
      ((buffer[7] & 0x0F) << 4 & 0xFF) | (buffer[8] >> 4),
      ((buffer[8] & 0x0F) << 4 & 0xFF) | (buffer[9] >> 4)
    ]
  receivedDeltas[0][3] = conv19bitToInt32(miniBuf)

  # Sample 2 - Channel 1
  miniBuf = [
      ((buffer[9] & 0x0F) >> 1),
      (buffer[9] << 7 & 0xFF) | (buffer[10] >> 1),
      (buffer[10] << 7 & 0xFF) | (buffer[11] >> 1)
    ]
  receivedDeltas[1][0] = conv19bitToInt32(miniBuf)

  # Sample 2 - Channel 2
  miniBuf = [
      ((buffer[11] & 0x01) << 2 & 0xFF) | (buffer[12] >> 6),
      (buffer[12] << 2 & 0xFF) | (buffer[13] >> 6),
      (buffer[13] << 2 & 0xFF) | (buffer[14] >> 6)
    ]
  receivedDeltas[1][1] = conv19bitToInt32(miniBuf)

  # Sample 2 - Channel 3
  miniBuf = [
      ((buffer[14] & 0x38) >> 3),
      ((buffer[14] & 0x07) << 5 & 0xFF) | ((buffer[15] & 0xF8) >> 3),
      ((buffer[15] & 0x07) << 5 & 0xFF) | ((buffer[16] & 0xF8) >> 3)
    ]
  receivedDeltas[1][2] = conv19bitToInt32(miniBuf)

  # Sample 2 - Channel 4
  miniBuf = [(buffer[16] & 0x07), buffer[17], buffer[18]]
  receivedDeltas[1][3] = conv19bitToInt32(miniBuf)

  return receivedDeltas;

def decompressDeltas18Bit(buffer):
  """
  Called to when a compressed packet is received.
  buffer: Just the data portion of the sample. So 19 bytes.
  return {Array} - An array of deltas of shape 2x4 (2 samples per packet and 4 channels per sample.)
  """ 
  if len(buffer) != 18:
    raise ValueError("Input should be 18 bytes long.")
  
  receivedDeltas = [[0, 0, 0, 0],[0, 0, 0, 0]]

  # Sample 1 - Channel 1
  miniBuf = [
      (buffer[0] >> 6),
      ((buffer[0] & 0x3F) << 2 & 0xFF) | (buffer[1] >> 6),
      ((buffer[1] & 0x3F) << 2 & 0xFF) | (buffer[2] >> 6)
    ]
  receivedDeltas[0][0] = conv18bitToInt32(miniBuf);

  # Sample 1 - Channel 2
  miniBuf = [
      (buffer[2] & 0x3F) >> 4,
      (buffer[2] << 4 & 0xFF) | (buffer[3] >> 4),
      (buffer[3] << 4 & 0xFF) | (buffer[4] >> 4)
    ]
  receivedDeltas[0][1] = conv18bitToInt32(miniBuf);

  # Sample 1 - Channel 3
  miniBuf = [
      (buffer[4] & 0x0F) >> 2,
      (buffer[4] << 6 & 0xFF) | (buffer[5] >> 2),
      (buffer[5] << 6 & 0xFF) | (buffer[6] >> 2)
    ]
  receivedDeltas[0][2] = conv18bitToInt32(miniBuf);

  # Sample 1 - Channel 4
  miniBuf = [
      (buffer[6] & 0x03),
      buffer[7],
      buffer[8]
    ]
  receivedDeltas[0][3] = conv18bitToInt32(miniBuf);

  # Sample 2 - Channel 1
  miniBuf = [
      (buffer[9] >> 6),
      ((buffer[9] & 0x3F) << 2 & 0xFF) | (buffer[10] >> 6),
      ((buffer[10] & 0x3F) << 2 & 0xFF) | (buffer[11] >> 6)
    ]
  receivedDeltas[1][0] = conv18bitToInt32(miniBuf);

  # Sample 2 - Channel 2
  miniBuf = [
      (buffer[11] & 0x3F) >> 4,
      (buffer[11] << 4 & 0xFF) | (buffer[12] >> 4),
      (buffer[12] << 4 & 0xFF) | (buffer[13] >> 4)
    ]
  receivedDeltas[1][1] = conv18bitToInt32(miniBuf);

  # Sample 2 - Channel 3
  miniBuf = [
      (buffer[13] & 0x0F) >> 2,
      (buffer[13] << 6 & 0xFF) | (buffer[14] >> 2),
      (buffer[14] << 6 & 0xFF) | (buffer[15] >> 2)
    ]
  receivedDeltas[1][2] = conv18bitToInt32(miniBuf);

  # Sample 2 - Channel 4
  miniBuf = [
      (buffer[15] & 0x03),
      buffer[16],
      buffer[17]
    ]
  receivedDeltas[1][3] = conv18bitToInt32(miniBuf);

  return receivedDeltas;