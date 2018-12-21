import cv2
import os
import numpy

import sys
import datetime
import time

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

import imutils
from imutils.video import FPS

from threading import Thread

class WebcamVideoStream:
  def __init__(self, resolution=(640, 480), framerate=32, src=0):
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






# VideoStream
stream = WebcamVideoStream().start()
time.sleep(2.0)

fps = FPS().start()

# Draw Options
#font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 32)
#font = ImageFont.truetype("arial.bold", 15)

#For Mac
font = ImageFont.truetype('/Library/Fonts/Arial.ttf', 16)

prediction = "No Label"

while True:
  # Capture frame-by-frame
  frame = stream.read()
  frame = cv2.flip(frame, 1)

  # Load array as an image
  img = Image.fromarray(frame)
  draw = ImageDraw.Draw(img)

  # Run inference with edgetpu
  #ans = engine.DetectWithImage(img, threshold=0.05, relative_coord=False, top_k=10)




  fps.update()
  fps.stop()
  current_fps = '{:.2f}'.format(fps.fps())
  text = 'Frames / Second: {}'.format(current_fps)

  draw.rectangle(((7,30),(200,70)), fill=None, outline='red')
  draw.text((10,40), text=text, font=font, align='center', fill=(0,0,255,128))

  # Display the resulting frame
  cv2.imshow('Video', numpy.array(img))

  fps.update()
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

# When everything is done, release the capture
cv2.destroyAllWindows()
stream.stop()
