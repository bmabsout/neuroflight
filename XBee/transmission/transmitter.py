import time
from crccheck.crc import Crc16Mcrf4XX
import serial
import ctypes

ser = serial.Serial('/dev/ttyUSB4', 115200)  # open first serial port
#print ser.portstr       # check which port was really used

def block_crc(block):
    crc = Crc16Mcrf4XX().calc(block)
    return ctypes.c_uint16(crc)

def listToString(s): 
    # initialize an empty string
    str1 = "" 
    # traverse in the string  
    for ele in s:
        str1 += chr(ele)
    split_strings = []
    n = 8
    for index in range(0, len(str1), n):
        split_strings.append(str1[index : index + n])

    #print(split_strings)    
    # return string  
    return split_strings 


model_tflite =  b'\x1c\x00\x00\x00\x54\x46\x4c\x33\x14\x00\x20\x00\x04\x00\x08\x00\x0c\x00\x10\x00\x14\x00\x00\x00\x18\x00\x1c\x00\x14\x00\x00\x00\x03\x00\x00\x00\x18\x00\x00\x00\x1c\x00\x00\x00\x90\x00\x00\x00\x1c\x00\x00\x00\x38\x00\x00\x00\x30\x00\x00\x00\x01\x00\x00\x00\x38\x01\x00\x00\x01\x00\x00\x00\x98\x00\x00\x00\x06\x00\x00\x00\x74\x03\x00\x00\x70\x03\x00\x00\x98\x02\x00\x00\x70\x01\x00\x00\x64\x03\x00\x00\x3c\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x0c\x00\x00\x00\x08\x00\x0c\x00\x04\x00\x08\x00\x08\x00\x00\x00\x08\x00\x00\x00\x05\x00\x00\x00\x13\x00\x00\x00\x6d\x69\x6e\x5f\x72\x75\x6e\x74\x69\x6d\x65\x5f\x76\x65\x72\x73\x69\x6f\x6e\x00\xb6\xfd\xff\xff\x04\x00\x00\x00\x10\x00\x00\x00\x31\x2e\x35\x2e\x30\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x0f\x00\x00\x00\x4d\x4c\x49\x52\x20\x43\x6f\x6e\x76\x65\x72\x74\x65\x64\x2e\x00\x00\x00\x0e\x00\x18\x00\x04\x00\x08\x00\x0c\x00\x10\x00\x14\x00\x0e\x00\x00\x00\x14\x00\x00\x00\x24\x00\x00\x00\x28\x00\x00\x00\x2c\x00\x00\x00\x30\x00\x00\x00\x04\x00\x00\x00\x84\x02\x00\x00\x1c\x02\x00\x00\xa4\x01\x00\x00\x80\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x03\x00\x00\x00\x01\x00\x00\x00\x1c\x00\x00\x00\x04\x00\x00\x00\x6d\x61\x69\x6e\x00\x00\x0e\x00\x14\x00\x00\x00\x08\x00\x0c\x00\x07\x00\x10\x00\x0e\x00\x00\x00\x00\x00\x00\x08\x18\x00\x00\x00\x0c\x00\x00\x00\x04\x00\x00\x00\x90\xfd\xff\xff\x01\x00\x00\x00\x03\x00\x00\x00\x03\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x01\x00\x00\x00\x0c\x00\x10\x00\x0b\x00\x00\x00\x0c\x00\x04\x00\x0c\x00\x00\x00\x09\x00\x00\x00\x00\x00\x00\x09\x01\x00\x00\x00\x1c\xfe\xff\xff\x14\x00\x00\x00\x04\x00\x00\x00\x18\x00\x00\x00\x30\x00\x00\x00\x20\x00\x00\x00\x02\x00\x00\x00\x01\x00\x00\x00\x04\x00\x00\x00\x08\x00\x00\x00\x49\x64\x65\x6e\x74\x69\x74\x79\x00\x00\x00\x00\x02\x00\x00\x00\xff\xff\xff\xff\x04\x00\x00\x00\x08\xfe\xff\xff\xe2\xfe\xff\xff\x04\x00\x00\x00\xd0\x00\x00\x00\x31\x2e\x87\x31\xf6\x87\x5e\x31\xec\xa7\xb5\x2f\x1a\x1e\x05\xb1\xa9\x0b\xb5\x30\xa8\xe8\xa2\x2e\xc1\x6e\x47\x31\x0b\x71\x14\xb1\xe5\x93\x90\x30\x67\x8d\x81\x2f\x6d\x66\x1f\xb0\xad\x33\x05\x30\x68\x46\xd4\x30\xd4\xf2\xb1\xb1\x98\x12\xa3\xb1\xe6\x95\x60\x31\x1a\xcb\x2a\x31\x14\x06\xc2\xb1\x37\xb2\xdb\xae\x2e\x54\x99\xb1\x86\xa0\xaf\x31\x58\x23\x08\xb1\x0a\x6b\xc8\xaf\xc0\x90\xf9\x2f\xde\xb3\x47\xb1\x80\x1d\x72\xb1\xce\xa7\x69\xae\x3b\x5e\x96\xaf\x96\xc2\x87\x2d\xa0\x73\x2b\x2f\x9f\x0b\xce\xae\xca\xf1\x36\xad\x60\xbd\x6a\xaf\x6f\x94\x95\x2f\x70\x20\xec\xae\xf4\xd8\x46\xaf\x5a\x0c\xe4\x2d\x8c\x5e\x38\xaf\x72\x6e\xa1\x2e\x66\xb0\x00\xb0\x07\x07\x47\xb1\xe6\xd2\x8e\xaf\x54\xf8\x02\x31\x48\x91\x63\xb0\x9e\xb9\xdb\xae\x9f\xff\x39\xb1\x07\x89\x5a\x31\x69\x06\xa6\xb0\x5a\x66\xba\x30\x52\x97\xb1\x2f\x6a\x07\xfb\xb0\xff\xb4\x81\xb0\x9a\xff\xff\xff\x10\x00\x00\x00\x03\x00\x00\x00\x14\x00\x00\x00\x2c\x00\x00\x00\x02\x00\x00\x00\x04\x00\x00\x00\x0d\x00\x00\x00\x12\x00\x00\x00\x6d\x6f\x64\x65\x6c\x2f\x64\x65\x6e\x73\x65\x2f\x4d\x61\x74\x4d\x75\x6c\x00\x00\x04\x00\x06\x00\x04\x00\x00\x00\x00\x00\x06\x00\x08\x00\x04\x00\x06\x00\x00\x00\x04\x00\x00\x00\x10\x00\x00\x00\xe6\x7e\x2d\x38\xf9\xc9\x81\xb8\x0a\x83\x68\xb6\x0f\xb9\xcc\x3d\x00\x00\x0e\x00\x14\x00\x04\x00\x00\x00\x08\x00\x0c\x00\x10\x00\x0e\x00\x00\x00\x10\x00\x00\x00\x02\x00\x00\x00\x10\x00\x00\x00\x3c\x00\x00\x00\x01\x00\x00\x00\x04\x00\x00\x00\x2b\x00\x00\x00\x6d\x6f\x64\x65\x6c\x2f\x64\x65\x6e\x73\x65\x2f\x42\x69\x61\x73\x41\x64\x64\x2f\x52\x65\x61\x64\x56\x61\x72\x69\x61\x62\x6c\x65\x4f\x70\x2f\x72\x65\x73\x6f\x75\x72\x63\x65\x00\xa8\xff\xff\xff\x14\x00\x18\x00\x04\x00\x00\x00\x08\x00\x0c\x00\x10\x00\x00\x00\x00\x00\x14\x00\x14\x00\x00\x00\x14\x00\x00\x00\x01\x00\x00\x00\x18\x00\x00\x00\x2c\x00\x00\x00\x1c\x00\x00\x00\x02\x00\x00\x00\x01\x00\x00\x00\x0d\x00\x00\x00\x07\x00\x00\x00\x69\x6e\x70\x75\x74\x5f\x31\x00\x02\x00\x00\x00\xff\xff\xff\xff\x0d\x00\x00\x00\xfc\xff\xff\xff\x04\x00\x04\x00\x04\x00\x00\x00'
model_tflite2 = b'\x1c\x00\x00\x00\x54\x46\x4c\x33\x14\x00\x20\x00\x04\x00\x08\x00\x0c\x00\x10\x00\x14\x00\x00\x00\x18\x00\x1c\x00\x14\x00\x00\x00\x03\x00\x00\x00\x18\x00\x00\x00\x1c\x00\x00\x00\x90\x00\x00\x00\x1c\x00\x00\x00\x38\x00\x00\x00\x30\x00\x00\x00\x01\x00\x00\x00\x38\x01\x00\x00\x01\x00\x00\x00\x98\x00\x00\x00\x06\x00\x00\x00\x74\x03\x00\x00\x70\x03\x00\x00\x98\x02\x00\x00\x70\x01\x00\x00\x64\x03\x00\x00\x3c\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x0c\x00\x00\x00\x08\x00\x0c\x00\x04\x00\x08\x00\x08\x00\x00\x00\x08\x00\x00\x00\x05\x00\x00\x00\x13\x00\x00\x00\x6d\x69\x6e\x5f\x72\x75\x6e\x74\x69\x6d\x65\x5f\x76\x65\x72\x73\x69\x6f\x6e\x00\xb6\xfd\xff\xff\x04\x00\x00\x00\x10\x00\x00\x00\x31\x2e\x35\x2e\x30\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x0f\x00\x00\x00\x4d\x4c\x49\x52\x20\x43\x6f\x6e\x76\x65\x72\x74\x65\x64\x2e\x00\x00\x00\x0e\x00\x18\x00\x04\x00\x08\x00\x0c\x00\x10\x00\x14\x00\x0e\x00\x00\x00\x14\x00\x00\x00\x24\x00\x00\x00\x28\x00\x00\x00\x2c\x00\x00\x00\x30\x00\x00\x00\x04\x00\x00\x00\x84\x02\x00\x00\x1c\x02\x00\x00\xa4\x01\x00\x00\x80\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x03\x00\x00\x00\x01\x00\x00\x00\x1c\x00\x00\x00\x04\x00\x00\x00\x6d\x61\x69\x6e\x00\x00\x0e\x00\x14\x00\x00\x00\x08\x00\x0c\x00\x07\x00\x10\x00\x0e\x00\x00\x00\x00\x00\x00\x08\x18\x00\x00\x00\x0c\x00\x00\x00\x04\x00\x00\x00\x90\xfd\xff\xff\x01\x00\x00\x00\x03\x00\x00\x00\x03\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x01\x00\x00\x00\x0c\x00\x10\x00\x0b\x00\x00\x00\x0c\x00\x04\x00\x0c\x00\x00\x00\x09\x00\x00\x00\x00\x00\x00\x09\x01\x00\x00\x00\x1c\xfe\xff\xff\x14\x00\x00\x00\x04\x00\x00\x00\x18\x00\x00\x00\x30\x00\x00\x00\x20\x00\x00\x00\x02\x00\x00\x00\x01\x00\x00\x00\x04\x00\x00\x00\x08\x00\x00\x00\x49\x64\x65\x6e\x74\x69\x74\x79\x00\x00\x00\x00\x02\x00\x00\x00\xff\xff\xff\xff\x04\x00\x00\x00\x08\xfe\xff\xff\xe2\xfe\xff\xff\x04\x00\x00\x00\xd0\x00\x00\x00\xa0\x05\xda\x2c\xd2\xf0\xcd\x2d\x0c\xe5\x02\xac\x79\x30\x8d\x2b\x99\x80\xd3\x2d\xc5\xba\x7b\xad\xf9\xf3\x51\x2d\xc2\xb8\xf8\x2c\x49\x96\xef\xac\x96\x68\x0f\x2d\x50\x33\xe5\xab\x4f\xa3\x12\xad\x55\x53\xdf\xad\x27\x23\x03\x96\x08\x1c\x25\x94\x28\x2b\xd9\x94\xbf\xbe\x63\x95\x38\xa4\xce\x94\xa0\x4e\x41\x14\xaf\x3a\x39\x95\x98\x47\x18\x94\xa8\xff\x7f\x95\xa7\x42\x3a\x14\x1d\x2d\xcc\x94\xa2\x3e\x42\x95\x8e\xd1\x8d\x95\x73\x16\xb6\x11\x90\x5a\x29\x11\x80\x24\x2a\x8f\x80\x6e\xf0\x8d\x22\xa8\x44\x91\x80\x25\xbf\x0e\x6d\xca\x5c\x91\x6c\x2e\xd3\x90\xa4\x93\xa3\x11\xfd\xeb\x6d\x91\x67\x80\x0b\x11\x14\xd7\xe9\x91\xec\x52\xc8\x10\x90\x4f\x19\x93\x93\xbd\x9d\x92\x8d\x94\xce\x12\x5c\xef\xd2\x92\xf6\x9e\x89\x13\x24\x6a\x32\x92\xd4\xad\x9a\x12\x4c\x83\xa2\x13\x7c\xa9\xed\x92\x4c\xaa\x86\x93\xba\x9a\x47\x12\x7e\x6f\x9e\x13\x6c\x35\xff\x90\x9a\xff\xff\xff\x10\x00\x00\x00\x03\x00\x00\x00\x14\x00\x00\x00\x2c\x00\x00\x00\x02\x00\x00\x00\x04\x00\x00\x00\x0d\x00\x00\x00\x12\x00\x00\x00\x6d\x6f\x64\x65\x6c\x2f\x64\x65\x6e\x73\x65\x2f\x4d\x61\x74\x4d\x75\x6c\x00\x00\x04\x00\x06\x00\x04\x00\x00\x00\x00\x00\x06\x00\x08\x00\x04\x00\x06\x00\x00\x00\x04\x00\x00\x00\x10\x00\x00\x00\xc9\xcc\xcc\x3d\x33\xc1\xda\x97\x74\xb3\xb1\x13\x58\x38\xf7\x14\x00\x00\x0e\x00\x14\x00\x04\x00\x00\x00\x08\x00\x0c\x00\x10\x00\x0e\x00\x00\x00\x10\x00\x00\x00\x02\x00\x00\x00\x10\x00\x00\x00\x3c\x00\x00\x00\x01\x00\x00\x00\x04\x00\x00\x00\x2b\x00\x00\x00\x6d\x6f\x64\x65\x6c\x2f\x64\x65\x6e\x73\x65\x2f\x42\x69\x61\x73\x41\x64\x64\x2f\x52\x65\x61\x64\x56\x61\x72\x69\x61\x62\x6c\x65\x4f\x70\x2f\x72\x65\x73\x6f\x75\x72\x63\x65\x00\xa8\xff\xff\xff\x14\x00\x18\x00\x04\x00\x00\x00\x08\x00\x0c\x00\x10\x00\x00\x00\x00\x00\x14\x00\x14\x00\x00\x00\x14\x00\x00\x00\x01\x00\x00\x00\x18\x00\x00\x00\x2c\x00\x00\x00\x1c\x00\x00\x00\x02\x00\x00\x00\x01\x00\x00\x00\x0d\x00\x00\x00\x07\x00\x00\x00\x69\x6e\x70\x75\x74\x5f\x31\x00\x02\x00\x00\x00\xff\xff\xff\xff\x0d\x00\x00\x00\xfc\xff\xff\xff\x04\x00\x04\x00\x04\x00\x00\x00'




# print(type(model_tflite_string))
# model_tflite_bytes = bytes(model_tflite_string, encoding="ASCII")
# model_tflite_bytes = list(map(lambda val: val.to_bytes(1, 'big'), model_tflite))


#Transmission of the first NN
print("\n\nTransmitting the first NN.")
length  = ctypes.c_uint16(len(model_tflite))
len_lsb = bytearray(length)[0]
len_msb = bytearray(length)[1]
crc     = block_crc(model_tflite)
crc_lsb = bytearray(crc)[0]
crc_msb = bytearray(crc)[1]
# print(length)
# print(crc)

ser.write(len_lsb)
ser.write(len_msb)
ser.write(crc_lsb)
ser.write(crc_msb)
ser.flush()
time.sleep(0.1)

for s in model_tflite:
    ser.write(s)
    #print(s)
    #ser.flush()
    time.sleep(0.01)

print("Transmission done.")

################################################################
time.sleep(10)
################################################################



#Transmission of the second NN
print("Transmitting the second NN")
length  = ctypes.c_uint16(len(model_tflite2))
len_lsb = bytearray(length)[0]
len_msb = bytearray(length)[1]
crc     = block_crc(model_tflite2)
crc_lsb = bytearray(crc)[0]
crc_msb = bytearray(crc)[1]
# print(length)
# print(crc)

ser.write(len_lsb)
ser.write(len_msb)
ser.write(crc_lsb)
ser.write(crc_msb)
ser.flush()
time.sleep(0.1)

for s in model_tflite2:
    ser.write(s)
    #print(s)
    #ser.flush()
    time.sleep(0.01)

print("Transmission done.")

# ser.close()             # close port
