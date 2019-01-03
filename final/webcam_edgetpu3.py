
import cv2
import os
import numpy

import sys
import datetime
import time

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from edgetpu.detection.engine import DetectionEngine

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
#Alternate function to load labels
def load_labels(filename):
  my_labels = []
  input_file = open(filename, 'r')
  for l in input_file:
    my_labels.append(l.strip())
  return my_labels

# Specify the model, image, and labels
orientation_model_path = './tflite_model/aiy_can_orientation_v2_edgetpu.tflite'
orientation_labels = ReadLabelFile('./tflite_model/can_orientation_v1_edgetpu.txt')
#labels = load_labels('./tflite_model/can_orientation_v1_edgetpu.txt')

presence_model_path = './tflite_model/detect_can_presence/aiy_2018-12-24_model_edgetpu.tflite'
presence_labels = ReadLabelFile('./tflite_model/detect_can_presence/can_presence_labels.txt')


# Initialize the engine
orientation_engine = ClassificationEngine(orientation_model_path)
presence_engine = ClassificationEngine(presence_model_path)

# VideoStream
stream = WebcamVideoStream().start()
time.sleep(2.0)

fps = FPS().start()

# Draw Options
font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 32)

#prediction = "No Label"

while True:
  # Capture frame-by-frame
  frame = stream.read()
  frame = cv2.flip(frame, 1)

  # Load array as an image
  img = Image.fromarray(frame)
  draw = ImageDraw.Draw(img)

  # Run inference with edgetpu

  orientation_prediction = "No Label"
  presence_prediction = "can_not_detected"

  print("Presence",presence_engine.ClassifyWithImage(img, threshold = 0.91, top_k=1))

  for result in presence_engine.ClassifyWithImage(img, threshold = 0.85, top_k=1):
    #print ('---------------------------')
    presence_prediction = presence_labels[result[0]]
  #  orientation_prediction = orientation_labels[result[0]]
    #score = result[2]
    #print ('Score : ', result[2])

  #for result in orientation_engine.ClassifyWithImage(img, threshold = 0.55, top_k=1):
  #  #print ('---------------------------')
  #  orientation_prediction = orientation_labels[result[0]]
  #  #score = result[2]
  #  #print ('Score : ', result[2])

  
  text = presence_prediction
  draw.text((0,10), text=text, font=font, fill='blue')

  fps.update()
  fps.stop()
  text = orientation_prediction
  draw.text((0,40), text=text, font=font, fill='blue')

 
  fps.update()
  fps.stop()
  current_fps = '{:.2f}'.format(fps.fps())
  text = 'Frames / Second: {}'.format(current_fps)
  draw.text((0,80), text=text, font=font, fill='blue')

  # Display the resulting frame
  cv2.imshow('Video', numpy.array(img))

  fps.update()
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

# When everything is done, release the capture
cv2.destroyAllWindows()
stream.stop()
