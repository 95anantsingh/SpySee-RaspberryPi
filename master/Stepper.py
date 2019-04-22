

## Author- ANANT SINGH
## Instructions for 6 Wire Stepper Motor
## Color--     ||  Yellow || Orange ||   Blue   ||   Pink   ||  Brown  ||   Red   ||
## Reference--       A1         B1         A2          B2
## BCM No.--         4          23         17          24
## Board No.--       7          16         11          18


import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BCM)

enable_pin = 18
coil_A_1_pin = 4
coil_A_2_pin = 17
coil_B_1_pin = 23
coil_B_2_pin = 24

##GPIO.setup(enable_pin, GPIO.OUT)
try:
    GPIO.setup(coil_A_1_pin, GPIO.OUT)
    GPIO.setup(coil_A_2_pin, GPIO.OUT)
    GPIO.setup(coil_B_1_pin, GPIO.OUT)
    GPIO.setup(coil_B_2_pin, GPIO.OUT)
except:
    RuntimeWarning
##GPIO.output(enable_pin, 1)


def forward(min,max, steps):
    delay= max
    for i in range(0, steps):
        setStep(1, 0, 1, 0)
        time.sleep(delay)
        setStep(0, 1, 1, 0)
        time.sleep(delay)
        setStep(0, 1, 0, 1)
        time.sleep(delay)
        setStep(1, 0, 0, 1)
        time.sleep(delay)
        if delay > min and i%100==99:
           delay -= 0.00001
           print(delay)


def backwards(min,max, steps):
    delay= max
    for i in range(0, steps):
        setStep(1, 0, 0, 1)
        time.sleep(delay)
        setStep(0, 1, 0, 1)
        time.sleep(delay)
        setStep(0, 1, 1, 0)
        time.sleep(delay)
        setStep(1, 0, 1, 0)
        time.sleep(delay)
        if delay > min:
            delay -= 0.001
            print(delay)


def setStep(w1, w2, w3, w4):
    GPIO.output(coil_A_1_pin, w1)
    GPIO.output(coil_A_2_pin, w2)
    GPIO.output(coil_B_1_pin, w3)
    GPIO.output(coil_B_2_pin, w4)


while True:
    maxDelay = input("Max Delay between steps (milliseconds)?")
    minDelay = input("Min Delay between steps (milliseconds)?")
    steps = input("How many steps forward? ")
    forward(float(minDelay)/ 1000.0,float(maxDelay)/1000.0, int(steps))
    steps = input("How many steps backwards? ")
    backwards(int(minDelay) / 1000.0,int(maxDelay)/1000.0, int(steps))