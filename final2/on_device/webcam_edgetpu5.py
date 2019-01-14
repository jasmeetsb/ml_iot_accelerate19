
import cv2
import os
import numpy
import sys
import datetime
import time
import csv
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import time
import datetime
import json
from edgetpu.classification.engine import ClassificationEngine

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
orientation_model_path = './aiy_can_orientation_v3_edgetpu.tflite'
orientation_labels = ReadLabelFile(
    './orientation_labels.txt')

presence_model_path = './aiy_can_presence_v3_edgetpu.tflite'
presence_labels = ReadLabelFile(
    './presence_labels.txt')


# Initialize the engine
orientation_engine = ClassificationEngine(orientation_model_path)
presence_engine = ClassificationEngine(presence_model_path)

# VideoStream
stream = WebcamVideoStream().start()
time.sleep(2.0)

fps = FPS().start()

# Draw Options
font = ImageFont.truetype(
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 14)


#Variables for can counter
previous_status = 1
cnt = 0
last_detection_time=0

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
    # capture mac adress for identity
    from uuid import getnode as get_mac
    mac = get_mac()

    # Run inference with edgetpu

    orientation_prediction = "No Label"
    presence_prediction = "Can not detected"

    presence_result = presence_engine.ClassifyWithImage(
        img, threshold=0.91, top_k=1)

    #print("Presence", st, presence_result)

    if not presence_result:
        previous_status = 1

    for result in presence_result:
        presence_prediction = presence_labels[result[0]]

        # Counter
        if((previous_status > result[0]) and ((time.time()-last_detection_time)>2)):
     #       print('New can detected')
            cnt = cnt+1
        previous_status = result[0]

        if(presence_prediction == 'Can detected'):
            #Record timestamp for last can detection
            last_detection_time=time.time()

            #Detect Can's orientation
            for result2 in orientation_engine.ClassifyWithImage(img, threshold=0.75, top_k=1):
                #result2= orientation_engine.ClassifyWithImage(img, threshold = 0.55, top_k=1)
               # print(result2, st)
              
               data = (st,mac,result[0],result2[0],result2[1],cnt)
               with open('data.csv', 'a',newline='') as f:
                   writer = csv.writer(f)
                   writer.writerow(data)
                  # write_to_csv(data)
                   sys.stdout.flush()
                   print(data)
                          
               orientation_prediction = orientation_labels[result2[0]]
            #score = result[2]
            #print ('Score : ', result[2])

    text = presence_prediction
    draw.rectangle(((0,0),(230,80)), fill='white', outline='black')
    draw.text((5, 5), text=text, font=font, fill='blue')

    fps.update()
    fps.stop()
    text = 'Can Orientation: '+orientation_prediction
    draw.text((5, 20), text=text, font=font, fill='blue')

    text = 'Can counter: '+str(cnt)
    draw.text((5, 35), text=text, font=font, fill='blue')

    fps.update()
    fps.stop()
    current_fps = '{:.2f}'.format(fps.fps())
    text = 'Frames / Second: {}'.format(current_fps)
    draw.text((5, 55), text=text, font=font, fill='blue')

    # Display the resulting frame
    cv2.imshow('Video', numpy.array(img))

    fps.update()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cv2.destroyAllWindows()
stream.stop()