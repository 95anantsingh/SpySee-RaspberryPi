import socket
from threading import Thread
import time

from gpiozero import DigitalOutputDevice
from gpiozero import PWMOutputDevice
import pigpio

import ultra_sonic as us

from threading import Lock

import RPi.GPIO as GPIO
import cv2
import io
import picamera
import logging
import socketserver
from threading import Condition
import serial

from imutils.video import FPS
from multiprocessing import Process
from multiprocessing import Queue
import numpy as np
import argparse
import imutils

IP = "192.10.9.96"
motionCtrlPort = 2110
camCtrlPort = 2111
specialPort = 2222
emergencyPort = 3333
videoPort = 1108
txMotionData = 0
txCamData = 0
txSpecialData = 0
txEmergencyData = 0
rxMotionData = 0
rxCamData = 90120
rxSpecialData = 0
rxEmergencyData = 0
output = ""
camAngle = 0
headAngle = 0
detections = None
ob = False
fd = False

Resolution = '753X424'
Framerate = 20

stopVideoSteamer = False
stopObjectDetecter = False

stopMotionDataUpdater = False
stopCamDataUpdater = False
stopSpecialDataUpdater = False

stopUltraSonicDataUpdater = False
stopEmergencyDataUpdater = False

stopSpecialController = False
stopMotionController = False
stopCamController = False

stopGpsDataUpdater = False
txUltraSonicData = 0
txGPSData = 0


def videoStreamer():
    global output, Framerate, Resolution
    from http import server
    while stopVideoSteamer is not True:
        print("video streamer started while")

        class StreamingOutput(object):
            def __init__(self):
                self.lock = Lock()
                self.frame = io.BytesIO()
                self.clients = []

            def write(self, buf):
                died = []
                if buf.startswith(b'\xff\xd8'):
                    # New frame, send old frame to all connected clients
                    size = self.frame.tell()
                    if size > 0:
                        self.frame.seek(0)
                        data = self.frame.read(size)
                        self.frame.seek(0)
                        with self.lock:
                            for client in self.clients:
                                try:
                                    client.wfile.write(b'--FRAME\r\n')
                                    client.send_header('Content-Type', 'image/jpeg')
                                    client.send_header('Content-Length', size)
                                    client.end_headers()
                                    client.wfile.write(data)
                                    client.wfile.write(b'\r\n')
                                except Exception as e:
                                    died.append(client)
                                    print(e)
                self.frame.write(buf)
                if died:
                    self.remove_clients(died)

            def flush(self):
                with self.lock:
                    for client in self.clients:
                        client.wfile.close()

            def add_client(self, client):
                print('Adding streaming client %s:%d' % client.client_address)
                with self.lock:
                    self.clients.append(client)

            def remove_clients(self, clients):
                with self.lock:
                    for client in clients:
                        try:
                            print('Removing streaming client %s:%d' % client.client_address)
                            self.clients.remove(client)
                        except ValueError:
                            pass  # already removed

        class StreamingHandler(server.BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == '/live.mjpg':
                    self.close_connection = False
                    self.send_response(200)
                    self.send_header('Age', 0)
                    self.send_header('Cache-Control', 'no-cache, private')
                    self.send_header('Pragma', 'no-cache')
                    self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=--FRAME')
                    self.end_headers()
                    output.add_client(self)
                else:
                    self.send_error(404)
                    self.end_headers()

        class StreamingServer(socketserver.ThreadingMixIn, server.HTTPServer):
            pass

        print('\nStreaming Started')
        with picamera.PiCamera(resolution=Resolution, framerate=Framerate) as camera:
            output = StreamingOutput()
            camera.start_recording(output, format='mjpeg')
            try:
                address = ('', videoPort)
                server = StreamingServer(address, StreamingHandler)
                server.serve_forever()
            finally:
                camera.stop_recording()


def objectDetector():
    from http import server
    global videoPort

    while stopObjectDetecter is not True:
        print("object detector started while")

        def classify_frame(net, inputQueue, outputQueue):
            # keep looping
            while True:
                # check to see if there is a frame in our input queue
                if not inputQueue.empty():
                    # grab the frame from the input queue, resize it, and
                    # construct a blob from it
                    frame = inputQueue.get()
                    frame = cv2.resize(frame, (760, 420))
                    blob = cv2.dnn.blobFromImage(frame, 0.007843,
                                                 (760, 420), 127.5)

                    # set the blob as input to our deep learning object
                    # detector and obtain the detections
                    net.setInput(blob)
                    detections = net.forward()

                    # write the detections to the output queue
                    outputQueue.put(detections)

        # construct the argument parse and parse the arguments
        ap = argparse.ArgumentParser()
        ap.add_argument("-p", "--prototxt",
                        help="path to Caffe 'deploy' prototxt file",
                        default='MobileNetSSD_deploy.prototxt.txt')
        ap.add_argument("-m", "--model",
                        help="path to Caffe pre-trained model",
                        default='MobileNetSSD_deploy.caffemodel')
        ap.add_argument("-c", "--confidence", type=float, default=0.2,
                        help="minimum probability to filter weak detections")
        args = vars(ap.parse_args())

        # initialize the list of class labels MobileNet SSD was trained to
        # detect, then generate a set of bounding box colors for each class
        CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
                   "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                   "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                   "sofa", "train", "tvmonitor"]
        COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

        # load our serialized model from disk
        print("[INFO] loading model...")
        net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

        # initialize the input queue (frames), output queue (detections),
        # and the list of actual detections returned by the child process
        inputQueue = Queue(maxsize=1)
        outputQueue = Queue(maxsize=1)

        # construct a child process *indepedent* from our main process of
        # execution
        print("[INFO] starting process...")
        p = Process(target=classify_frame, args=(net, inputQueue,
                                                 outputQueue,))
        p.daemon = True
        p.start()

        class StreamingOutput(object):
            def __init__(self):
                self.frame = None
                self.buffer = io.BytesIO()
                self.condition = Condition()

            def write(self, buf):
                if buf.startswith(b'\xff\xd8'):
                    # New frame, copy the existing buffer's content and notify all
                    # clients it's available
                    self.buffer.truncate()
                    with self.condition:
                        self.frame = self.buffer.getvalue()
                        self.condition.notify_all()
                    self.buffer.seek(0)
                return self.buffer.write(buf)

        class StreamingHandler(server.BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == '/live.mjpg':
                    self.send_response(200)
                    self.send_header('Age', 0)
                    self.send_header('Cache-Control', 'no-cache, private')
                    self.send_header('Pragma', 'no-cache')
                    self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
                    self.end_headers()
                    try:
                        while True:
                            with output.condition:
                                output.condition.wait()
                                fps = FPS().start()
                                frame = output.frame
                                data = np.fromstring(frame, dtype=np.uint8)
                                frame = cv2.imdecode(data, 1)
                                frame = imutils.resize(frame, width=760)
                                (fH, fW) = frame.shape[:2]

                                # if the input queue *is* empty, give the current frame to
                                # classify
                                if inputQueue.empty():
                                    inputQueue.put(frame)

                                # if the output queue *is not* empty, grab the detections
                                if not outputQueue.empty():
                                    global detections
                                    detections = outputQueue.get()

                                # check to see if our detectios are not None (and if so, we'll
                                # draw the detections on the frame)
                                if detections is not None:
                                    # loop over the detections
                                    for i in np.arange(0, detections.shape[2]):
                                        # extract the confidence (i.e., probability) associated
                                        # with the prediction
                                        confidence = detections[0, 0, i, 2]

                                        # filter out weak detections by ensuring the `confidence`
                                        # is greater than the minimum confidence
                                        if confidence < args["confidence"]:
                                            continue

                                        # otherwise, extract the index of the class label from
                                        # the `detections`, then compute the (x, y)-coordinates
                                        # of the bounding box for the object
                                        idx = int(detections[0, 0, i, 1])
                                        dims = np.array([fW, fH, fW, fH])
                                        box = detections[0, 0, i, 3:7] * dims
                                        (startX, startY, endX, endY) = box.astype("int")

                                        # draw the prediction on the frame
                                        label = "{}: {:.2f}%".format(CLASSES[idx],
                                                                     confidence * 100)
                                        cv2.rectangle(frame, (startX, startY), (endX, endY),
                                                      COLORS[idx], 2)
                                        y = startY - 15 if startY - 15 > 15 else startY + 15
                                        cv2.putText(frame, label, (startX, y),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

                                        # show the output frame

                                        # update the FPS counter
                                fps.update()
                                fps.stop()
                            # stop the timer and display FPS information

                            r, frame = cv2.imencode(".jpg", frame)

                            self.wfile.write(b'--FRAME\r\n')
                            self.send_header('Content-Type', 'image/jpeg')
                            self.send_header('Content-Length', len(frame))
                            self.end_headers()
                            self.wfile.write(frame)
                            self.wfile.write(b'\r\n')
                    except Exception as e:
                        logging.warning(
                            'Removed streaming client %s: %s',
                            self.client_address, str(e))
                else:
                    self.send_error(404)
                    self.end_headers()

        class StreamingServer(socketserver.ThreadingMixIn, server.HTTPServer):
            allow_reuse_address = True
            daemon_threads = True

        with picamera.PiCamera(resolution='426X240', framerate=40)as camera:
            output = StreamingOutput()
            camera.start_recording(output, format='mjpeg')
            try:
                address = ('', videoPort)
                server = StreamingServer(address, StreamingHandler)
                server.serve_forever()
            finally:
                camera.stop_recording()


def objectStreamer():
    from http import server
    global videoPort

    def classify_frame(net, inputQueue, outputQueue):
        # keep looping
        while True:
            # check to see if there is a frame in our input queue
            if not inputQueue.empty():
                # grab the frame from the input queue, resize it, and
                # construct a blob from it
                frame = inputQueue.get()
                frame = cv2.resize(frame, (640, 360))
                blob = cv2.dnn.blobFromImage(frame, 0.007843,
                                             (640, 360), 127.5)

                # set the blob as input to our deep learning object
                # detector and obtain the detections
                net.setInput(blob)
                detections = net.forward()

                # write the detections to the output queue
                outputQueue.put(detections)

    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--prototxt",
                    help="path to Caffe 'deploy' prototxt file",
                    default='MobileNetSSD_deploy.prototxt.txt')
    ap.add_argument("-m", "--model",
                    help="path to Caffe pre-trained model",
                    default='MobileNetSSD_deploy.caffemodel')
    ap.add_argument("-c", "--confidence", type=float, default=0.2,
                    help="minimum probability to filter weak detections")
    args = vars(ap.parse_args())

    # initialize the list of class labels MobileNet SSD was trained to
    # detect, then generate a set of bounding box colors for each class
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]
    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

    # load our serialized model from disk
    print("[INFO] loading model...")
    net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

    # initialize the input queue (frames), output queue (detections),
    # and the list of actual detections returned by the child process
    inputQueue = Queue(maxsize=1)
    outputQueue = Queue(maxsize=1)
    detections = None

    # construct a child process *indepedent* from our main process of
    # execution
    print("[INFO] starting process...")
    p = Process(target=classify_frame, args=(net, inputQueue,
                                             outputQueue,))
    p.daemon = True
    p.start()

    class StreamingOutput(object):
        def __init__(self):
            self.frame = None
            self.buffer = io.BytesIO()
            self.condition = Condition()

        def write(self, buf):
            if buf.startswith(b'\xff\xd8'):
                # New frame, copy the existing buffer's content and notify all
                # clients it's available
                self.buffer.truncate()
                with self.condition:
                    self.frame = self.buffer.getvalue()
                    self.condition.notify_all()
                self.buffer.seek(0)
            return self.buffer.write(buf)

    class StreamingHandler(server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == '/live.mjpg':
                self.send_response(200)
                self.send_header('Age', 0)
                self.send_header('Cache-Control', 'no-cache, private')
                self.send_header('Pragma', 'no-cache')
                self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
                self.end_headers()
                try:
                    while True:
                        with output.condition:
                            output.condition.wait()
                            fps = FPS().start()
                            frame = output.frame
                            data = np.fromstring(frame, dtype=np.uint8)
                            frame = cv2.imdecode(data, 1)

                            frame = imutils.resize(frame, width=640)

                            (fH, fW) = frame.shape[:2]

                            # if the input queue *is* empty, give the current frame to
                            # classify
                            if inputQueue.empty():
                                inputQueue.put(frame)

                            # if the output queue *is not* empty, grab the detections
                            if not outputQueue.empty():
                                global detections
                                detections = outputQueue.get()

                            # check to see if our detectios are not None (and if so, we'll
                            # draw the detections on the frame)
                            if detections is not None:
                                # loop over the detections
                                for i in np.arange(0, detections.shape[2]):
                                    # extract the confidence (i.e., probability) associated
                                    # with the prediction
                                    confidence = detections[0, 0, i, 2]

                                    # filter out weak detections by ensuring the `confidence`
                                    # is greater than the minimum confidence
                                    if confidence < args["confidence"]:
                                        continue

                                    # otherwise, extract the index of the class label from
                                    # the `detections`, then compute the (x, y)-coordinates
                                    # of the bounding box for the object
                                    idx = int(detections[0, 0, i, 1])
                                    dims = np.array([fW, fH, fW, fH])
                                    box = detections[0, 0, i, 3:7] * dims
                                    (startX, startY, endX, endY) = box.astype("int")

                                    # draw the prediction on the frame
                                    label = "{}: {:.2f}%".format(CLASSES[idx],
                                                                 confidence * 100)
                                    cv2.rectangle(frame, (startX, startY), (endX, endY),
                                                  COLORS[idx], 2)
                                    y = startY - 15 if startY - 15 > 15 else startY + 15
                                    cv2.putText(frame, label, (startX, y),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

                                    # show the output frame

                                    # update the FPS counter
                            fps.update()
                            fps.stop()
                            # stop the timer and display FPS information

                            r, frame = cv2.imencode(".jpg", frame)

                        self.wfile.write(b'--FRAME\r\n')
                        self.send_header('Content-Type', 'image/jpeg')
                        self.send_header('Content-Length', len(frame))
                        self.end_headers()
                        self.wfile.write(frame)
                        self.wfile.write(b'\r\n')
                except Exception as e:
                    logging.warning(
                        'Removed streaming client %s: %s',
                        self.client_address, str(e))
            else:
                self.send_error(404)
                self.end_headers()

    class StreamingServer(socketserver.ThreadingMixIn, server.HTTPServer):
        allow_reuse_address = True
        daemon_threads = True

    with picamera.PiCamera(resolution='640x360', framerate=25)as camera:
        output = StreamingOutput()
        camera.start_recording(output, format='mjpeg')
        try:
            address = ('', 1108)
            server = StreamingServer(address, StreamingHandler)
            server.serve_forever()
        finally:
            camera.stop_recording()


def switcher():
    global rxMotionData, stopObjectDetecter, stopVideoSteamer
    if rxMotionData == 90000 or rxMotionData == 0:
        stopObjectDetecter = True
        stopVideoSteamer = False
        videoStreamerThread.start()
    else:
        stopObjectDetecter = False
        stopVideoSteamer = True
        objectDetectorThread.start()


def threadController():
    while True:
        switcherThread.start()


def motionDataUpdater():
    global IP, motionCtrlPort, rxMotionData
    print('\nmotionDataUpdater Started')
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((IP, motionCtrlPort))
    while stopMotionDataUpdater is not True:
        msg = str(txMotionData)
        sendBytes = msg.encode('utf-8')
        # print("TXMotionData: %f" % txMotionData)
        data, addr = sock.recvfrom(1024)
        sock.sendto(sendBytes, addr)
        dataString = data.decode('utf-8')
        rxMotionData = int(dataString)
        # print("RXMotionData: %f" % rxMotionData)


def camDataUpdater():
    global IP, camCtrlPort, rxCamData
    print('camDataUpdater Started')
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((IP, camCtrlPort))
    while stopCamDataUpdater is not True:
        msg = str(txCamData)
        sendBytes = msg.encode('utf-8')
        data, addr = sock.recvfrom(1024)
        sock.sendto(sendBytes, addr)
        dataString = data.decode('utf-8')
        rxCamData = int(dataString)
        # print(rxCamData)


def specialDataUpdater():
    global IP, specialPort, rxSpecialData
    # print('specialDataUpdater Started')
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((IP, specialPort))
    while stopSpecialDataUpdater is not True:
        msg = str(txSpecialData)
        sendBytes = msg.encode('utf-8')
        data, addr = sock.recvfrom(1024)
        sock.sendto(sendBytes, addr)
        dataString = data.decode('utf-8')
        rxSpecialData = dataString
        # print(rxSpecialData)


def emergencyDataUpdater():
    global IP, emergencyPort, rxEmergencyData, stopEmergencyDataUpdater
    print('emergencyDataUpdater Started')
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((IP, emergencyPort))
    while stopEmergencyDataUpdater is not True:
        msg = str(txEmergencyData)
        sendBytes = msg.encode('utf-8')
        data, addr = sock.recvfrom(1024)
        sock.sendto(sendBytes, addr)
        dataString = data.decode('utf-8')
        rxEmergencyData = int(dataString)


def gpsDataUpdater():
    global txGPSData
    print('GPS')
    try:
        ser = serial.Serial("/dev/serial0")
    except PermissionError:
        print('Permission Denied... Retrying...')
        from time import sleep
        sleep(10)
        try:
            ser = serial.Serial("/dev/serial0")
        except PermissionError:
            print('Permission Denied. Error!')

    def convertToDegrees(raw_value):
        decimal_value = raw_value / 100.00
        degrees = int(decimal_value)
        position = degrees + ((decimal_value - int(decimal_value)) / 0.6)
        position = "%.4f" % position
        return position

    def convertTime(raw_time):
        raw_time = raw_time + 53000
        if raw_time > 240000:
            raw_time = raw_time - 240000
        return raw_time

    while True:
        received_data = str(ser.readline())                      # read NMEA string received
        GPGGA_data_available = received_data.find("$GPGGA,")     # check for NMEA GPGGA string
        if (GPGGA_data_available > 0):
            GPGGA_buffer = received_data.split("$GPGGA,", 1)[1]  # store data coming after "$GPGGA," string
            NMEA_buff = (GPGGA_buffer.split(','))                # store comma separated data in buffer

            dilution = float(NMEA_buff[8])
            if dilution != 99.99:
                dlat = NMEA_buff[2]
                dlon = NMEA_buff[4]
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
                    print('No Data, Dilution: ', dilution)
            else:
                print('Antenna not connected properly, Dilution: ', dilution)

def ultraSonicDataUpdater():
    # GPIO Mode (BOARD / BCM)
    global txUltraSonicData


    def distanceA(triggerPin, echoPin):
        global distanceA
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(triggerPin, GPIO.OUT)
        GPIO.setup(echoPin, GPIO.IN)
        while True:
            GPIO.output(triggerPin, True)
            time.sleep(0.00001)
            GPIO.output(triggerPin, False)
            StartTime = time.time()
            StopTime = time.time()
            while GPIO.input(echoPin) == 0:
                StartTime = time.time()
            while GPIO.input(echoPin) == 1:
                StopTime = time.time()
            TimeElapsed = StopTime - StartTime
            distanceA = int((TimeElapsed * 34300) / 2)
            #print(distanceA)

    def distanceB(triggerPin, echoPin):
        global distanceB
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(triggerPin, GPIO.OUT)
        GPIO.setup(echoPin, GPIO.IN)
        while True:
            GPIO.output(triggerPin, True)
            time.sleep(0.00001)
            GPIO.output(triggerPin, False)
            StartTime = time.time()
            StopTime = time.time()
            while GPIO.input(echoPin) == 0:
                StartTime = time.time()
            while GPIO.input(echoPin) == 1:
                StopTime = time.time()
            TimeElapsed = StopTime - StartTime
            distanceB = int((TimeElapsed * 34300) / 2)
            print(distanceB)

    # set GPIO Pins
    triggerA = 27    # 15
    echoA = 22       # 13
    triggerB = 20    # 38
    echoB = 21       # 40

    #ultraA = Thread(target= distanceA(triggerA, echoA))
    #ultraB = Thread(target= distanceB(triggerB, echoB))
    print('print started1')
   # ultraA.start()
    print('print started2')
   # ultraB.start()
    print('print started')

    # txUltraSonicData = distanceA * 1000 + distanceB

    while True:
        print(distanceA,'  ', distanceB)
        #print(distanceB)
        time.sleep(0.1)


def motionController():
    global txMotionData, camAngle, headAngle

    leftPowerPin = 12  # 32IN1 - Forward Drive
    leftDirPin = 16  # 36IN2 - Reverse Drive
    rightPowerPin = 24  # 18 IN1 - Forward Drive
    rightDirPin = 23  # 16IN2 - Reverse Drive

    leftPower = PWMOutputDevice(leftPowerPin, True, 0, 1000)
    leftDir = DigitalOutputDevice(leftDirPin, True, False)
    rightPower = PWMOutputDevice(rightPowerPin, True, 0, 1000)
    rightDir = DigitalOutputDevice(rightDirPin, True, False)

    # print('\nmotionController Started')
    while stopMotionController is not True:

        power = int(rxMotionData % 1000)
        # if power!= 0:
        angle = int(rxMotionData / 1000)
        power = (power + 1) / 256
        if angle in range(80, 101) or angle in range(260, 281):
            rPower = power
            lPower = power
        elif angle in range(101, 181):
            rPower = power
            lPower = power * ((180 - angle) / 90)
        elif angle in range(181, 261):
            rPower = power
            lPower = power * ((angle - 180) / 90)
        elif angle in range(0, 80):
            rPower = power * ((angle) / 90)
            lPower = power
        else:
            rPower = power * ((359 - angle) / 90)
            lPower = power

        if lPower < 0.1: lPower = 0
        if rPower < 0.1: rPower = 0

        if angle in range(0, 181):
            leftDir.off()
            rightDir.off()
            leftPower.value = lPower
            rightPower.value = rPower
        else:
            leftDir.on()
            rightDir.on()
            leftPower.value = lPower
            rightPower.value = rPower
        txMotionData = int(lPower + rPower)
        time.sleep(0.04)
        # print(" rxMD:%d" % rxMotionData, " A:%d" % angle," P:%.2f" % power,
        #      " lPower:%.2f" % lPower, " rPower:%.2f" % rPower)

        # print(" rxMD:%d" % rxMotionData, " txMD:%d" % txMotionData,
        #      " rxCD:%d" % rxCamData, " txCD:%d" % txCamData,
        #     " rxSD:%d" % rxSpecialData, " txSD:%d" % txSpecialData,
        #     " rxED:%d" % rxEmergencyData, " txED:%d" % txEmergencyData)


def camController():
    global txCamData, camAngle, headAngle, lastCamData, pi
    lastCamData = 909090909
    camPin = 26
    headPin = 17
    pi = pigpio.pi()
    pi.set_mode(camPin, pigpio.OUTPUT)
    pi.set_mode(headPin, pigpio.OUTPUT)

    def cam(an, pin):
        an = round((6.666667 * an) + (0.012346 * an * an) + 500)
        pi.set_servo_pulsewidth(pin, an)

    cam(120, camPin)
    cam(0, headPin)

    while True:
        camAngle = int(rxCamData % 1000)  # end digits  xxxabc  abc is data
        headAngle = int(rxCamData / 1000)  # begin digits abcxxx
        if rxCamData is not lastCamData:
            lastCamData = rxCamData
            cam(camAngle, camPin)
            cam(headAngle, headPin)
            # print("rxCamData: %d" % rxCamData,"camAngle: %d" % camAngle, "headAngle: %d" % headAngle)


def specialController():
    global txSpecialData
    print('specialController Started')
    while stopSpecialController is not True:
        txSpecialData += 1
        time.sleep(0.1)
        # if rxSpecialData!=0:
        # print("rxSpecialData: %d" % rxSpecialData)


def emergencyController():
    global txEmergencyData, stopMotionController, stopSpecialController, stopCamController, stopMotionDataUpdater, \
        stopCamDataUpdater, stopUltraSonicDataUpdater, stopGyroDataUpdater, stopGpsDataUpdater, stopSpecialDataUpdater, \
        stopEmergencyDataUpdater, stopVideoSteamer, stopObjectDetecter
    global ob, fd

    lastTxEmergencyData = 890

    motionControllerStopCommand = 101
    motionControllerStartCommand = 102

    specialControllerStopCommand = 103
    specialControllerStartCommand = 104

    camControllerStopCommand = 105
    camControllerStartCommand = 106

    motionDataUpdaterStopCommand = 107
    motionDataUpdaterStartCommand = 108

    camDataUpdaterStopCommand = 109
    camDataUpdaterStartCommand = 110

    ultraSonicDataUpdateStopCommand = 111
    ultraSonicDataUpdateStartCommand = 112

    gyroDataUpdaterStopCommand = 113
    gyroDataUpdaterStartCommand = 114

    gpsDataUpdaterStopCommand = 115
    gpsDataUpdaterStartCommand = 116

    specialDataUpdaterStopCommand = 117
    specialDataUpdaterStartCommand = 118

    videoSteamerStopCommand = 119
    videoSteamerStartCommand = 120

    objectDetectorStopCommand = 121
    objectDetectorStartCommand = 122

    emergencyDataUpdaterStopCommand = 123
    emergencyDataUpdaterStartCommand = 124

    obStartCommand = 125
    obStopCommand = 126

    fdStartCommand = 127
    fdStopCommand = 128

    print('emergencyController Started')
    while True:
        if txEmergencyData is not lastTxEmergencyData:
            lastTxEmergencyData = txEmergencyData
            if txEmergencyData is motionControllerStopCommand:
                stopMotionController = True
            # elif txEmergencyData is motionControllerStartCommand:
            #    stopMotionController = False
            # elif txEmergencyData is specialControllerStopCommand:
            #    stopSpecialController = True
            # elif txEmergencyData is specialControllerStartCommand:
            #    stopSpecialController = False
            # elif txEmergencyData is camControllerStopCommand:
            #    stopCamController = True
            # elif txEmergencyData is camControllerStartCommand:
            #    stopCamController = False
            # elif txEmergencyData is motionDataUpdaterStopCommand:
            #    stopMotionDataUpdater = True
            # elif txEmergencyData is motionDataUpdaterStartCommand:
            #    stopMotionDataUpdater = False
            # elif txEmergencyData is camDataUpdaterStopCommand:
            #    stopCamDataUpdater = True
            # elif txEmergencyData is camDataUpdaterStartCommand:
            #    stopCamDataUpdater = False
            # elif txEmergencyData is ultraSonicDataUpdateStopCommand:
            #    stopUltraSonicDataUpdater = True
            # elif txEmergencyData is ultraSonicDataUpdateStartCommand:
            #   stopUltraSonicDataUpdater = False
            # elif txEmergencyData is gyroDataUpdaterStopCommand:
            #    stopGyroDataUpdater = True
            # elif txEmergencyData is gyroDataUpdaterStartCommand:
            #    stopGyroDataUpdater = False
            # elif txEmergencyData is gpsDataUpdaterStopCommand:
            #    stopGpsDataUpdater = True
            # elif txEmergencyData is gpsDataUpdaterStartCommand:
            #    stopGpsDataUpdater = False
            # elif txEmergencyData is specialDataUpdaterStopCommand:
            #    stopSpecialDataUpdater = True
            # elif txEmergencyData is specialDataUpdaterStartCommand:
            #    stopSpecialDataUpdater = False
            # elif txEmergencyData is emergencyDataUpdaterStopCommand:
            #    stopEmergencyDataUpdater = True
            # elif txEmergencyData is emergencyDataUpdaterStartCommand:
            #    stopEmergencyDataUpdater = False
            # elif txEmergencyData is videoSteamerStopCommand:
            #    stopVideoSteamer = True
            # elif txEmergencyData is videoSteamerStartCommand:
            #    stopVideoSteamer = False
            # elif txEmergencyData is objectDetectorStopCommand:
            #    stopObjectDetecter = True
            # elif txEmergencyData is objectDetectorStartCommand:
            #    stopObjectDetecter = False
            elif txEmergencyData is obStartCommand:
                ob = True
            elif txEmergencyData is obStopCommand:
                ob = False
            elif txEmergencyData is fdStartCommand:
                fd = True
            elif txEmergencyData is fdStopCommand:
                fd = False


if __name__ == '__main__':
    try:
        print('\nStarting Threads...')
        motionDataUpdaterThread = Thread(target=motionDataUpdater)
        camDataUpdaterThread = Thread(target=camDataUpdater)
        specialDataUpdaterThread = Thread(target=specialDataUpdater)
        emergencyDataUpdaterThread = Thread(target=emergencyDataUpdater)

        motionControllerThread = Thread(target=motionController)
        camControllerThread = Thread(target=camController)
        specialControllerThread = Thread(target=specialController)
        emergencyControllerThread = Thread(target=emergencyController)

        videoStreamerThread = Thread(target=videoStreamer)
        objectDetectorThread = Thread(target=objectDetector)

        switcherThread = Thread(target=switcher)

        ultrasonicThread = Thread(target=ultraSonicDataUpdater)

        gpsDataUpdater = Thread(target=gpsDataUpdater)

        threadController = Thread(target=threadController)

        objectSteamerThread = Thread(target=objectStreamer)

        # switcherThread.start()
        # motionDataUpdaterThread.start()
        # camDataUpdaterThread.start()
        # specialDataUpdaterThread.start()
        # emergencyDataUpdaterThread.start()

        # motionControllerThread.start()
        # camControllerThread.start()
        # specialControllerThread.start()
        # emergency

        #ultrasonicThread.start()
        gpsDataUpdater.start()
        # threadController.start()
        # objectSteamerThread.start()
        # videoStreamerThread.start()
        print('\nStarted')
    except KeyboardInterrupt:
        print('\nInterrupted')
        GPIO.cleanup()
        pi.stop()
