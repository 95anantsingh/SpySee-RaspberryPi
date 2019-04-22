from gpiozero import AngularServo
from time import sleep

servo = AngularServo(17,min_angle=-90, max_angle=90, min_pulse_width=(0.5/1000),max_pulse_width=(2.3/1000))
servo.angle=0
sleep(1)
servo.angle= 45
while True:
  a=int(input('Enter Angle: '))
  servo.angle=a
