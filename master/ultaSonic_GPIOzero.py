from gpiozero import DistanceSensor
from time import sleep


if __name__ == '__main__':
    sensor = DistanceSensor(echo=21, trigger=20)
    while True:
        print('Distance: ', sensor.distance * 100,)
        sleep(0.1)