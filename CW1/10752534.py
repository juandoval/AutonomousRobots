#Start code
#Q1 Answer: 587
#Q2 Answer: 14
#Q3 Answer: 16th January 2018

import csv
from datetime import datetime, timezone

#Open the binary input file
input_file = open("binaryFileC_29.bin", 'rb')
output_file = open("10752534.csv", 'w', newline='')
writer = csv.writer(output_file)

def lookup_temp(value):
    if 0xA0 <= value <= 0xDF:
        return round(30.0 + (value - 0xA0) * 0.1, 1)
    else:
        return 0.0

#Read the first byte and loop as long as
#there is always another byte available
byte = input_file.read(1)

# initializing tracking variables
position = 0
frame=[]
previous_byte = b''
frame_count = 0
corrupt_count = 0
first_timestamp = None

while byte: # != b"", is not empty
    # print("Byte value is (hexidecimal): " + str(byte))
    # print("Byte value is (decimal): " + str(int.from_bytes(byte)))
    if position == 0:
        # searching for the start of frame
        if previous_byte == b'~' and byte == b'~':
            # Found start of frame, byte 1-2 "~~"
            frame = [ord(b'~'), ord(b'~')] # ord converts byte to int for the frame list
            position = 3 # position is now 3, since 1 and 2: ~~
        previous_byte = byte # updating previous byte for the next iteration

    elif position <= 25:
        # collect bytes 3-25 of the frame
        frame.append(int.from_bytes(byte, byteorder='big'))
        position += 1
    
    elif position == 26:
        # checksum, byte 26, frame is now complete
        frame.append(int.from_bytes(byte, byteorder='big'))
        
        if frame[7] != ord(b'P') or frame[16] != ord(b'T'):
            # validating the frame, byte 7 should be "P" and byte 16 should be "T", which is constant
            # therefore reset and search for the next frame
            print("invalid frame, searching for next frame")
            position = 0
            frame = []
        else:    
            # DECODE
            
            # HEADER
            sys_id = frame[2] # byte 3
            dest_id = frame[3] # byte 4
            comp_id = frame[4] # byte 5
            seq = frame[5] # byte 6
            msg_type = frame[6] # byte 7
            
            # PAYLOAD
            # byte 8 is "P" (payload start marker), skip it
            rpm = int.from_bytes(bytes([frame[8], frame[9]]), byteorder='big') # byte 9-10, big endian (MSB 1st)
            vlt = int.from_bytes(bytes([frame[10], frame[11]]), byteorder='big') # byte 11-12, big endian (MSB 1st)
            crt = int.from_bytes(bytes([frame[12], frame[13]]), byteorder='little', signed=True) # byte 13-14, little endian (LSB 1st)
            mos_tmp = lookup_temp(frame[14]) # byte 15, temperature of mosfet using the lookup table
            cap_tmp = lookup_temp(frame[15]) # byte 16, temperature of capacitor using the lookup table
            
            # TIMING
            # skip byte 17, which is "T"
            timestamp = int.from_bytes(bytes(frame[17:25]), byteorder='big') # byte 18-25, big endian (MSB 1st)
            
            # CHECKSUM
            checksum = frame[25] # byte 26
            
            # validate checksum, which is the sum of bytes 3-25 modulo 256
            calc_checksum = 255 - (sum(frame[0:25]) % 256)
            if calc_checksum != checksum:
                corrupt_count += 1
                print("frame is corrupt")

            # write decoded values to CSV in the required format
            writer.writerow(["~~", sys_id, dest_id, comp_id, seq, msg_type,
                             "P", rpm, vlt, crt, mos_tmp, cap_tmp,
                             "T", timestamp, checksum, ""])

            # store the first timestamp for Q3
            if first_timestamp is None:
                first_timestamp = timestamp

            # print the index of frame and the frame as a list of integers
            print(f"frame N{frame_count}" + str(frame))
            frame_count += 1
        
        # reset for next frame
        position = 0
        frame = []
        previous_byte = b''

    #Get the next byte from the file and repeat
    byte = input_file.read(1)


#Must be end of the file so close the file
print("End of file reached")
print(f"Q1: Total complete frames: {frame_count}")
print(f"Q2: Corrupt frames: {corrupt_count}")
# convert first timestamp from microseconds to a UTC datetime for the calendar date
print(f"Q3: Calendar date of messages: {datetime.fromtimestamp(first_timestamp / 1_000_000, timezone.utc)}")
input_file.close()
output_file.close()

#End of code
