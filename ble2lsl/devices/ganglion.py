"""Interfacing parameters for the OpenBCI Ganglion Board."""

import numpy as np
from pygatt import BLEAddressType
import struct

PARAMS = dict(
    manufacturer='OpenBCI',
    units='microvolts',
    ch_names=('A', 'B', 'C', 'D'),
    chunk_size=1,
    packet_dtypes=dict(index='uint:16', ch_value='uint:12'),
    ble=dict(
        address_type=BLEAddressType.random,
        service='fe84',
        receive={'handles': [25],
                 'last_handle': 25,
                 'uuids': ['2d30c082f39f4ce6923f3484ea480596']},
        send="2d30c083f39f4ce6923f3484ea480596",
        stream_on=b'b',
        stream_off=b's',
        disconnect="2d30c084f39f4ce6923f3484ea480596",
    ),
)

STREAM_PARAMS = dict(
    name='Ganglion',
    type='EEG',
    channel_count=4,
    nominal_srate=200,
    channel_format='float32'
)

INT_SIGN_BYTE = (b'\x00', b'\xff')

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
      self.last_channel_data = [0, 0, 0, 0]
      # 18bit data got here and then accelerometer with it
      self.last_accelerometer = [0, 0, 0]
      # when the board is manually set in the right mode (z to start, Z to stop), impedance will be measured. 4 channels + ref
      self.last_impedance = [0, 0, 0, 0, 0]
      self.scaling_output = scaling_output

      # TODO: order in terms of probability of receiving (faster)
      self.byte_id_ranges = {(0, 0): self.parse_raw,
                             (1, 100): self.parse18bit,
                             (101, 200): self.parse19bit,
                             (201, 205):self.parse_impedance,
                             (206, 207): self.print_ascii,
                             (208, -1): self.unknown_packet_warning}

  def process_packet(self, packet):
    """Parse incoming data packet.

    Calls the corresponding parsing function depending on packet format.
    """
    start_byte = packet[0]
    for r in self.byte_id_ranges:
        if start_byte >= r[0] and start_byte <= r[1]:
            self.byte_id_ranges[r](start_byte, packet[1:])
            break

  def push_sample(self, sample_id, chan_data, aux_data, imp_data):
    """ Add a sample to inner stack, setting ID and dealing with scaling if necessary. """
    if self.scaling_output:
        chan_data = np.array([chan_data]) * scale_fac_uVolts_per_count
        #aux_data = np.array(aux_data) * scale_fac_accel_G_per_count
    self.sample = [sample_id, chan_data.T]

  def update_packets_count(self, packet_id):
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
          print("Warning: dropped {} packets.".format(self.packets_dropped))

  def unknown_packet_warning(self, start_byte, packet):
      """Print if incoming byte ID is unknown."""
      print("Warning: unknown type of packet: {}".format(start_byte))

  def print_ascii(self, start_byte, packet):
      """Print verbose ASCII data.

      TODO:
          * optional log file
      """
      print("%\t" + str(packet))
      if start_byte == 207:
          print("$$$\n")

  def parse_raw(self, packet_id, packet):
    """Parse a raw uncompressed packet."""
    if len(packet) != 19:
      print('Wrong size for raw data, {} instead of 19 bytes'.format(len(data)))
      return

    # 4 channels of 24bits, take values one by one
    chan_data = [conv24bitsToInt(packet[i:i+3]) for i in range(0, 12, 3)]

    # save uncompressed raw channel for future use and append whole sample
    self.push_sample(packet_id, chan_data, self.last_accelerometer, self.last_impedance)
    self.last_channel_data = chan_data
    self.update_packets_count(packet_id)

  def parse19bit(self, packet_id, packet):
    """Parse a 19-bit compressed packet without accelerometer data."""
    if len(packet) != 19:
      print('Wrong size for 19-bit compression data, {} instead of 19 bytes'.format(len(data)))
      return
    packet_id -= 100
    # should get 2 by 4 arrays of uncompressed data
    deltas = decompressDeltas19Bit(packet)
    # the sample_id will be shifted
    delta_id = 1
    for delta in deltas:
      # convert from packet to sample id
      sample_id = (packet_id - 1) * 2 + delta_id
      # 19bit packets hold deltas between two samples
      # TODO: use more broadly numpy
      full_data = list(np.array(self.last_channel_data) - np.array(delta))
      # NB: aux data updated only in 18bit mode, send values here only to be consistent
      self.push_sample(sample_id, full_data, self.last_accelerometer, self.last_impedance)
      self.last_channel_data = full_data
      delta_id += 1
    self.update_packets_count(packet_id)

  def parse18bit(self, packet_id, packet):
    """ Dealing with "18-bit compression without Accelerometer" """
    if len(packet) != 19:
      print('Wrong size for 18-bit compression data, {} instead of 19 bytes'.format(len(data)))
      return

    # set appropriate accelerometer byte
    self.last_accelerometer[packet_id % 10 - 1] = conv8bitToInt8(packet[18])

    # deltas: should get 2 by 4 arrays of uncompressed data
    deltas = decompressDeltas18Bit(packet[:-1])
    # the sample_id will be shifted
    delta_id = 1
    for delta in deltas:
      # convert from packet to sample id
      sample_id = (packet_id - 1) * 2 + delta_id
      # 19bit packets hold deltas between two samples
      # TODO: use more broadly numpy
      full_data = list(np.array(self.last_channel_data) - np.array(delta))
      self.push_sample(sample_id, full_data, self.last_accelerometer, self.last_impedance)
      self.last_channel_data = full_data
      delta_id += 1
    self.update_packets_count(packet_id)

  def parse_impedance(self, packet_id, packet):
    """ Dealing with impedance data. packet: ASCII data. NB: will take few packet (seconds) to fill"""
    if packet[-2:] != "Z\n":
      print('Wrong format for impedance check, should be ASCII ending with "Z\\n"')

    # convert from ASCII to actual value
    imp_value = int(packet[:-2])
    # from 201 to 205 codes to the right array size
    self.last_impedance[packet_id - 201] = imp_value
    self.push_sample(packet_id - 200, self.last_channel_data,
                     self.last_accelerometer, self.last_impedance)

def conv24bitsToInt(unpacked):
  """Convert 24-bit data coded on 3 bytes to a proper integer."""
  if len(unpacked) != 3:
    raise ValueError("Input should be 3 bytes long.")

  # FIXME: quick'n dirty, unpack wants strings later on
  int_bytes = INT_SIGN_BYTE[unpacked[0] > 127] + struct.pack('3B', *unpacked)

  #unpack little endian(>) signed integer(i) (makes unpacking platform independent)
  int_unpacked = struct.unpack('>i', int_bytes)[0]

  return int_unpacked

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
    return (256 - byte) * (-1)
  else:
    return byte

def decompressDeltas19Bit(buffer):
  """Called to when a compressed packet is received.
  buffer: Just the data portion of the sample. So 19 bytes.
  return {Array} - An array of deltas of shape 2x4 (2 samples per packet and 4 channels per sample.)
  """
  if len(buffer) != 19:
    raise ValueError("Input should be 19 bytes long.")

  receivedDeltas = [[0, 0, 0, 0],
                    [0, 0, 0, 0]]

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
  """Called to when a compressed packet is received.
  buffer: Just the data portion of the sample. So 19 bytes.
  return {Array} - An array of deltas of shape 2x4 (2 samples per packet and 4 channels per sample.)
  """
  if len(buffer) != 18:
    raise ValueError("Input should be 18 bytes long.")

  receivedDeltas = [[0, 0, 0, 0],
                    [0, 0, 0, 0]]

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
