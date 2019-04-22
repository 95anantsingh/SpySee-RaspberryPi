txGyroData=0

stopGpsDataUpdater = False
stopGyroDataUpdater = False


def gyroDataUpdater():
    PWR_MGMT_1 = 0x6B
    SMPLRT_DIV = 0x19
    CONFIG = 0x1A
    GYRO_CONFIG = 0x1B
    INT_ENABLE = 0x38
    ACCEL_XOUT_H = 0x3B
    ACCEL_YOUT_H = 0x3D
    ACCEL_ZOUT_H = 0x3F
    GYRO_XOUT_H = 0x43
    GYRO_YOUT_H = 0x45
    GYRO_ZOUT_H = 0x47
    Gx = 0
    Gy = 0
    Gz = 0
    global txGyroData

    while stopGyroDataUpdater is not True:

        def MPU_Init():
            # write to sample rate register
            bus.write_byte_data(Device_Address, SMPLRT_DIV, 7)

            # Write to power management register
            bus.write_byte_data(Device_Address, PWR_MGMT_1, 1)

            # Write to Configuration register
            bus.write_byte_data(Device_Address, CONFIG, 0)

            # Write to Gyro configuration register
            bus.write_byte_data(Device_Address, GYRO_CONFIG, 24)

            # Write to interrupt enable register
            bus.write_byte_data(Device_Address, INT_ENABLE, 1)
        def read_raw_data(addr):
            # Accelero and Gyro value are 16-bit
            high = bus.read_byte_data(Device_Address, addr)
            low = bus.read_byte_data(Device_Address, addr + 1)

            # concatenate higher and lower value
            value = ((high << 8) | low)

            # to get signed value from mpu6050
            if (value > 32768):
                value = value - 65536
            return value

        bus = smbus.SMBus(1)  # or bus = smbus.SMBus(0) for older version boards
        Device_Address = 0x68  # MPU6050 device address

        MPU_Init()

        print(" Reading Data of Gyroscope and Accelerometer")

        while True:
            # Read Accelerometer raw value
            acc_x = read_raw_data(ACCEL_XOUT_H)
            acc_y = read_raw_data(ACCEL_YOUT_H)
            acc_z = read_raw_data(ACCEL_ZOUT_H)

            # Read Gyroscope raw value
            gyro_x = read_raw_data(GYRO_XOUT_H)
            gyro_y = read_raw_data(GYRO_YOUT_H)
            gyro_z = read_raw_data(GYRO_ZOUT_H)

            # Full scale range +/- 250 degree/C as per sensitivity scale factor
            Ax = acc_x / 16384.0
            Ay = acc_y / 16384.0
            Az = acc_z / 16384.0

            Gx = gyro_x / 131.0
            Gy = gyro_y / 131.0
            Gz = gyro_z / 131.0

            # print("Gx=%.2f" % Gx, u'\u00b0' + "/s", "\tGy=%.2f" % Gy, u'\u00b0' + "/s", "\tGz=%.2f" % Gz, u'\u00b0' + "/s",
            # "\tAx=%.2f g" % Ax, "\tAy=%.2f g" % Ay, "\tAz=%.2f g" % Az


            # gyroVariable = Gx1*(10**6) + Gy1*(10**3) + Gz1
            # return gyroVariable

            Sf = 2  # Range of negative values is between 200 and 400
            Gx1 = int(Gx)
            Gy1 = int(Gy)
            Gz1 = int(Gz)

            if (Gx1 < 0):
                Gx1 = abs(Gx1) + 500

            if (Gy1 < 0):
                Gy1 = abs(Gy1) + 500

            if (Gz1 < 0):
                Gz1 = abs(Gz1) + 500

            Ax1 = round(Ax, 2)
            Ay1 = round(Ay, 2)
            Az1 = round(Az, 2)
            if (Ax1 > 0):
                Ax1 = Ax1 * 100
            if (Ay1 > 0):
                Ay1 = Ay1 * 100
            if (Az1 > 0):
                Az1 = Az1 * 100
            if (Ax1 < 0):
                Ax1 = (abs(Ax1) + Sf) * 100
            if (Ay1 < 0):
                Ay1 = (abs(Ay1) + Sf) * 100
            if (Az1 < 0):
                Az1 = (abs(Az1) + Sf) * 100

            txGyroData = Gx1 * 10 ** 15 + Gy1 * 10 ** 12 + Gz1 * 10 ** 9 + int(Ax1 * (10 ** 6) + Ay1 * (10 ** 3) + Az1)

            # acceloVariable1 = Ax1*(10**3) + Ay1


            # print("\tAx=%.2f g" % Ax, "\tAy=%.2f g" % Ay, "\tAz=%.2f g" % Az,accelo1,gyro1)

            # print("Gx=%.2f" % Gx, u'\u00b0' + "/s", "\tGy=%.2f" % Gy, u'\u00b0' + "/s", "\tGz=%.2f" % Gz, u'\u00b0' + "/s",
            #     "\tAx=%.2f g" % Ax, "\tAy=%.2f g" % Ay, "\tAz=%.2f g" % Az,GyroData)
            #sleep(1)