import socket
from threading import Thread
import time
#from gpiozero import Motor
#from gpiozero import AngularServo
import io
import picamera
import socketserver
from threading import Lock
from http import server

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
    with picamera.PiCamera(resolution='640x360', framerate=23) as camera:
        output = StreamingOutput()
        camera.start_recording(output, format='mjpeg')
        try:
            address = ('',1108)
            server = StreamingServer(address, StreamingHandler)
            server.serve_forever()
        finally:
            camera.stop_recording()



def motionControler():
    print('\nmotionControler Started')
    while True:
        txMotionData=txMotionData+1
        time.sleep(0.01)
        if(rxMotionData!=0 and rxMotionData!=90000):
            print("MotionData: %d"%(rxMotionData))

def armControler():

    print('\narmControler Started')
    while True:
        txArmData=txArmData+1
        time.sleep(0.01)
        if(rxArmData!=0):
            print("ArmData: %d"%(rxArmData))

def camControler():
    print('\ncamControler Started')
    while True:
        txCamData=txCamData+1
        time.sleep(0.01)
        if(rxCamData!=0):
            print("CamData: %d"%(rxCamData))

def motionDataUpdater():
    global rxMotionData
    rxMotionData=0
    print('\nmotionDataUpdater Started')
    sock = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    sock.bind((IP, motionCtrlPort))
    while True:
        msg=str(txMotionData)
        sendBytes=msg.encode('utf-8')
        data, addr = sock.recvfrom(1024)
        sent=sock.sendto(sendBytes,addr)
        dataString=data.decode('utf-8')
        rxMotionData=int(dataString)

def armDataUpdater():
    global rxArmData
    rxArmData=0
    print('\narmDataUpdater Started')
    sock = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    sock.bind((IP, armCtrlPort))
    while True:
        msg=str(txArmData)
        sendBytes=msg.encode('utf-8')
        data, addr = sock.recvfrom(1024)
        sent=sock.sendto(sendBytes,addr)
        dataString=data.decode('utf-8')
        rxArmData=int(dataString)

def camDataUpdater():
    global rxCamData
    rxCamData=0
    print('\ncamDataUpdater Started')
    sock = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    sock.bind((IP, camCtrlPort))
    while True:
        msg=str(txCamData)
        sendBytes=msg.encode('utf-8')
        data, addr = sock.recvfrom(1024)
        sent=sock.sendto(sendBytes,addr)
        dataString=data.decode('utf-8')
        rxCamData=int(dataString)


if __name__=='__main__':
    global IP, motionCtrlPort, camCtrlPort, armCtrlPort, txMotionData, txArmData, txCamData
    IP="192.10.9.96"
    motionCtrlPort=2110
    armCtrlPort=2111
    camCtrlPort=2112
    txMotionData=0
    txArmData=0
    txCamData=0
    print('\nStarting Threads...')

    motionDataUpdaterThread = Thread(target = motionDataUpdater)
    armDataUpdaterThread = Thread(target = armDataUpdater)
    camDataUpdaterThread = Thread(target = camDataUpdater)

    motionControlerThread = Thread(target = motionControler)
    armControlerThread = Thread(target = armControler)
    camControlerThread = Thread(target = camControler)

    videoStreamerThread = Thread(target=videoStreamer)

    motionDataUpdaterThread.start()
    armDataUpdaterThread.start()
    camDataUpdaterThread.start()

    motionControlerThread.start()
    armControlerThread.start()
    camControlerThread.start()

     #videoStreamerThread.start()
