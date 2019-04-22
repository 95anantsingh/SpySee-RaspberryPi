import socket
from threading import Thread
import time
from gpiozero import Motor
from gpiozero import PWMOutputDevice
from gpiozero import AngularServo
import io
import picamera
import socketserver
from threading import Lock
from http import server

IP = "192.10.9.96"
motionCtrlPort = 2110
camCtrlPort = 2111
specialPort = 2222
emergencyPort = 3333
txMotionData=0
txCamData = 0
txSpecialData = 0
txEmergencyData = 0
rxMotionData = 0
rxCamData = 0
rxSpecialData = 0
rxEmergencyData = 0
output=""
camAngle = 0
headAngle = 0


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
                    pass # already removed
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
def videoStreamer():
    global output
    print('\nStreaming Started')
    with picamera.PiCamera(resolution='640x360', framerate=30) as camera:
        output = StreamingOutput()
        camera.start_recording(output, format='mjpeg')
        try:
            address = ('',1108)
            server = StreamingServer(address, StreamingHandler)
            server.serve_forever()
        finally:
            camera.stop_recording()


def motionDataUpdater():
    global IP, motionCtrlPort, rxMotionData
    print('\nmotionDataUpdater Started')
    sock = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    sock.bind((IP, motionCtrlPort))
    while True:
        msg=str(txMotionData)
        sendBytes=msg.encode('utf-8')
        #print("TXMotionData: %f" % txMotionData)
        data, addr = sock.recvfrom(1024)
        sock.sendto(sendBytes,addr)
        dataString=data.decode('utf-8')
        rxMotionData=int(dataString)
        #print("RXMotionData: %f" % rxMotionData)
def camDataUpdater():
    global IP, camCtrlPort, rxCamData
    print('camDataUpdater Started')
    sock = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    sock.bind((IP, camCtrlPort))
    while True:
        msg=str(txCamData)
        sendBytes=msg.encode('utf-8')
        data, addr = sock.recvfrom(1024)
        sock.sendto(sendBytes,addr)
        dataString=data.decode('utf-8')
        rxCamData=int(dataString)
def specialDataUpdater():
    global IP, specialPort, rxSpecialData
    print('specialDataUpdater Started')
    sock = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    sock.bind((IP, specialPort))
    while True:
        msg=str(txSpecialData)
        sendBytes=msg.encode('utf-8')
        data, addr = sock.recvfrom(1024)
        sock.sendto(sendBytes,addr)
        dataString=data.decode('utf-8')
        rxSpecialData=(dataString)
        print(rxSpecialData)
def emergencyDataUpdater():
    global IP, emergencyPort, rxEmergencyData
    print('emergencyDataUpdater Started')
    sock = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    sock.bind((IP, emergencyPort))
    while True:
        msg=str(txEmergencyData)
        sendBytes=msg.encode('utf-8')
        data, addr = sock.recvfrom(1024)
        sock.sendto(sendBytes,addr)
        dataString=data.decode('utf-8')
        rxEmergencyData=int(dataString)

def motionController():
    global txMotionData , camAngle, headAngle

    leftPositivePin = 26  # IN1 - Forward Drive
    leftNegativePin = 19  # IN2 - Reverse Drive
    rightPositivePin = 13  # IN1 - Forward Drive
    rightNegativePin = 6  # IN2 - Reverse Drive

    leftPositive = PWMOutputDevice(leftPositivePin, True, 0, 1000)
    leftNegative = PWMOutputDevice(leftNegativePin, True, 0, 1000)
    rightPositive = PWMOutputDevice(rightPositivePin, True, 0, 1000)
    rightNegative = PWMOutputDevice(rightNegativePin, True, 0, 1000)

    print('\nmotionController Started')
    while True:
        power = int(rxMotionData%1000)
       # if power!= 0:
        angle = int(rxMotionData / 1000)
        power = (power + 1) / 256
        if angle in range (80, 101) or angle in range (260, 281):
            rPower = power
            lPower = power
        elif angle in range (101, 181):
            rPower = power
            lPower = power * ((180-angle)/90)
        elif angle in range (181, 261):
            rPower = power
            lPower = power * ((angle - 180) / 90)
        elif angle in range(0, 80):
            rPower = power * ((angle)/90)
            lPower = power
        else:
            rPower = power * ((359-angle) / 90)
            lPower = power

        if lPower<0.1: lPower = 0
        if rPower<0.1: rPower = 0


        if angle in range (0, 181):
            leftNegative.value = 0
            rightNegative.value = 0
            leftPositive.value = lPower
            rightPositive.value = rPower
        else:
            leftPositive.value = 0
            rightPositive.value = 0
            leftNegative.value = lPower
            rightNegative.value = rPower
        txMotionData =int((lPower)+(rPower))
        #time.sleep(0.04)
        print("A:%.2f" % angle, " P:%.2f" % power, " lP:%.2f" % lPower, " rP:%0.2f" % rPower,
             " rxMD:%0.2f" % rxMotionData, " txMD:%d" % txMotionData,
              " Ca:%0.2f" % camAngle, " ha:%0.2f" % headAngle,
              " rxCD:%0.2f" % rxCamData, " txCD:%0.2f" % txCamData)
def camController():
    global txCamData, camAngle, headAngle
    camPin = 4
    headPin = 17
    print('camController Started')
    camServo = AngularServo(camPin, min_angle=0, max_angle=180, min_pulse_width=(0.5 / 1000),
                            max_pulse_width=(2.3 / 1000),frame_width=20/1000)
    headServo = AngularServo(headPin, min_angle=-90, max_angle=90, min_pulse_width=(0.5 / 1000),
                            max_pulse_width=(2.3 / 1000),frame_width=20 / 1000)
    camServo.angle = 90
    headServo.angle = 0
    time.sleep(1)

    while True:
        txCamData = 0
        #time.sleep(0.500)
        camAngle = int(rxCamData%1000) #end digits  xxxabc  abc is data
        headAngle = int(rxCamData/1000) #begin digits abcxxx
        #if rxCamData:
            #print("rxCamData: %d" % rxCamData)
        camServo.angle = camAngle
        headServo.angle = headAngle
def specialController():
    global txSpecialData
    print('specialController Started')
    while True:
        txSpecialData += 1
        time.sleep(0.1)
        #if rxSpecialData!=0:
        # print("rxSpecialData: %d" % rxSpecialData)
def emergencyController():
    global txEmergencyData
    print('emergencyController Started')
    while True:
        txEmergencyData += 1
        time.sleep(0.1)
        if rxEmergencyData!=0:
            print("rxEmergencyData: %d" % rxEmergencyData)



if __name__=='__main__':
    print('\nStarting Threads...')
    
    motionDataUpdaterThread = Thread(target = motionDataUpdater)
    camDataUpdaterThread = Thread(target = camDataUpdater)
    specialDataUpdaterThread = Thread(target=specialDataUpdater)
    emergencyDataUpdaterThread = Thread(target=emergencyDataUpdater)

    motionControllerThread = Thread(target = motionController)
    camControllerThread = Thread(target = camController)
    specialControllerThread = Thread(target = specialController)
    emergencyControllerThread = Thread(target = emergencyController)

    videoStreamerThread = Thread(target = videoStreamer)

    motionDataUpdaterThread.start()
    camDataUpdaterThread.start()
    specialDataUpdaterThread.start()
    emergencyDataUpdaterThread.start()
    
    motionControllerThread.start()
    camControllerThread.start()
    specialControllerThread.start()
    emergencyControllerThread.start()

    #videoStreamerThread.start()
