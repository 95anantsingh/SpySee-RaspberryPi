import pigpio
import time

pi = pigpio.pi()
pi.set_mode(17, pigpio.OUTPUT)

pi.set_servo_pulsewidth(17, 1500)
a=1500
while True:
    a= int(input('Enter Angle: '))
    a = round((6.666667 * a) + (0.01234568 * a *a) + 500)
    pi.set_servo_pulsewidth(17, a)


