
# Code to stream video at 192.10.9.96/live.mjpg
# Any browser can be used to recieve the stream if it is connected to the SpySee Hotspot.
# This code uses threads please close all the previous executions of this script before invoking a new one.


import socket
from threading import Thread
import time
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
    with picamera.PiCamera(resolution='640x360', framerate=30) as camera:
        output = StreamingOutput()
        camera.start_recording(output, format='mjpeg')
        try:
            address = ('',1108)
            server = StreamingServer(address, StreamingHandler)
            server.serve_forever()
        finally:
            camera.stop_recording()



if __name__=='__main__':
    global IP
    IP="192.10.9.96"

    print('\nStarting Threads...')

    videoStreamerThread = Thread(target=videoStreamer)
    videoStreamerThread.start()


#   To use Thread version uncomment the following

#if __name__=='__main__':
#    global IP
#    IP="192.10.9.96"
#
#    videoStreamer()
