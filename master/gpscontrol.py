

import serial   #import serial pacakge


#convert raw NMEA string into format
def convertToDegrees(raw_value):
    decimal_value = raw_value/100.00
    degrees = int(decimal_value)
    position = degrees + ((decimal_value - int(decimal_value))/0.6)
    position = "%.4f" %(position)
    return position
def convertTime(raw_time):
    raw_time = raw_time + 53000
    if raw_time >240000:
        raw_time = raw_time - 240000
    return raw_time

gpgga_info = "$GPGGA,"
ser = serial.Serial("/dev/serial0")              #Open port with baud rate
GPGGA_buffer = 0
NMEA_buff = 0

while True:
    received_data = (str)(ser.readline())                   #read NMEA string received
    GPGGA_data_available = received_data.find(gpgga_info)   #check for NMEA GPGGA string
    if (GPGGA_data_available>0):
        GPGGA_buffer = received_data.split("$GPGGA,",1)[1]  #store data coming after "$GPGGA," string
        NMEA_buff = (GPGGA_buffer.split(','))               #store comma separated data in buffer

        dilution = float(NMEA_buff[8])
        if dilution != 99.9:
            dlat = NMEA_buff[2]
            dlon = NMEA_buff[4]  # extract data out of NMEA packet
            status = NMEA_buff[5]
            satellites = NMEA_buff[6]
            altitude = NMEA_buff[8]

            try:
                time = convertTime(int(float(NMEA_buff[0])))
                latitude = convertToDegrees(float(NMEA_buff[1]))
                longitude = convertToDegrees(float(NMEA_buff[3]))

                # txGPSData = latitude * (10 ** 4) * (10 ** 6) + longitude * (10 ** 4)

                print(
                    '\n', 'Time: ', time,
                    '\n', 'Latitude: ', latitude, dlat,
                    '\n', 'Longitude: ', longitude, dlon,
                    '\n', 'Altitude: ', altitude, 'meters',
                    '\n', 'Status: ', status,
                    '\n', 'Satellites: ', satellites,
                    '\n', 'Dilution: ', dilution
                )
                print("------------------------------------------------------------\n")

            except ValueError:
                print('No Data, Dilution: ', dilution)  # get time, latitude, longitude
        else:
            print('Antenna not connected properly, Dilution: ', dilution)                                     #get time, latitude, longitude