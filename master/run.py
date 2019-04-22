import io
import picamera
import logging
import socketserver
from threading import Condition
from http import server
from imutils.video import VideoStream
from imutils.video import FPS
from multiprocessing import Process
from multiprocessing import Queue
import numpy as np
import argparse
import imutils
import time
import cv2



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

					r, frame = cv2.imencode(".jpg",frame)

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

with picamera.PiCamera(resolution='760x420', framerate=30)as camera:
	output = StreamingOutput()
	camera.start_recording(output, format='mjpeg')
	try:
		address = ('', 1108)
		server = StreamingServer(address, StreamingHandler)
		server.serve_forever()
	finally:
		camera.stop_recording()
