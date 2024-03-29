import cv2
import os
import numpy
import sys
import datetime
import time
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import time

import time
import datetime

from edgetpu.classification.engine import ClassificationEngine

#For Object Detection
from edgetpu.detection.engine import DetectionEngine
import imutils
from imutils.video import FPS
from threading import Thread


class WebcamVideoStream:
    def __init__(self, resolution=(640, 480), framerate=32, src=-1):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        self.stream.set(cv2.CAP_PROP_FPS, framerate)
        self.grabbed, self. frame = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return

            self.grabbed, self.frame = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True


# Function to read labels from text files.
def ReadLabelFile(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    ret = {}
    for line in lines:
        pair = line.strip().split(':', maxsplit=1)
        ret[int(pair[0])] = pair[1].strip()
    return ret


# Specify the model, image, and labels
#orientation_model_path = './tflite_model/detect_coke_can_orientation/models_on-device_ICN6775484298763384503_jan20_edgetpu.tflite'
#orientation_labels = ReadLabelFile('./tflite_model/detect_coke_can_orientation/orientation_labels.txt')

#presence_model_path = './tflite_model/detect_coke_can_presence/models_on-device_ICN2207431524157418399_jan20_edgetpu.tflite'
#presence_labels = ReadLabelFile('./tflite_model/detect_coke_can_presence/presence_labels.txt')

###Object Detection
obj_model_path = './tflite_model/mobilenet_ssd_v2_face_quant_postprocess_edgetpu2.tflite'
obj_labels = {0: 'face', 1: 'background'}


# Initialize the engine
#orientation_engine = ClassificationEngine(orientation_model_path)
#presence_engine = ClassificationEngine(presence_model_path)

##Object Detection
print("Begin  creating engine instance")
obj_engine = DetectionEngine(obj_model_path)
print("Done..")



# VideoStream
stream = WebcamVideoStream().start()
time.sleep(2.0)



# Draw Options
font = ImageFont.truetype(
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 14)




while True:
    # Capture frame-by-frame
    frame = stream.read()
    frame = cv2.flip(frame, 1)

    # Load array as an image
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)

    # time and date functions to capture part of the telemetry to IoT Core
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')

    # Run inference with edgetpu
    ans = obj_engine.DetectWithImage(img, threshold=0.05, relative_coord=False, top_k=5)


    if ans:
      for coke_can in ans:
        draw.rectangle(coke_can.bounding_box.flatten().tolist(), outline='red')





    
    # Display the resulting frame
    cv2.imshow('Video', numpy.array(img))


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cv2.destroyAllWindows()
stream.stop()
