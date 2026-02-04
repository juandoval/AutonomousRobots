#Start of skeleton code

#Open the binary input file
input_file = open("<<put your file path here>.bin", 'rb')

#Read the first byte and loop as long as
#there is always another byte available
byte = input_file.read(1)
while byte:
    print("Byte value is (hexidecimal): " + str(byte))
    print("Byte value is (decimal): " + str(int.from_bytes(byte)))

    #
    #
    # add decoding method here
    #
    #

    #Get the next byte from the file and repeat
    byte = input_file.read(1)


#Must be end of the file so close the file
print("End of file reached")
input_file.close()

#End of skeleton code
