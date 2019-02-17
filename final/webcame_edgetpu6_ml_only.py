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
import json
import csv
# capture mac adress for identity
from uuid import getnode as get_mac
mac = get_mac()


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
orientation_model_path = './aiy_can_orientation_final_edgetpu.tflite'
orientation_labels = ReadLabelFile(
    './orientation_labels.txt')

presence_model_path = './aiy_can_presence_final_edgetpu.tflite'
presence_labels = ReadLabelFile(
    './presence_labels.txt')

###Object Detection
obj_model_path = './aiy_obj_det_final_edgetpu.tflite'
#obj_labels = {0: 'face', 1: 'background'}
obj_labels = {1: 'coke_can'}

# Initialize the engine
orientation_engine = ClassificationEngine(orientation_model_path)
presence_engine = ClassificationEngine(presence_model_path)

##Object Detection
print("Begin  creating engine instance")
obj_engine = DetectionEngine(obj_model_path)
print("Done..")



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

    # Run inference with edgetpu
    ## Time the inference for prediction model
    #pres_start_time = time.time()
    ans = obj_engine.DetectWithImage(img, threshold=0.05, relative_coord=False, top_k=1)
    #pres_end_time = time.time()
    #pres_inference_time = pres_end_time - pres_start_time
    #print('Inference time:',pres_inference_time)

    if ans:
      for coke_can in ans:
        draw.rectangle(coke_can.bounding_box.flatten().tolist(), outline='red')

    orientation_prediction = "No Label"
    presence_prediction = "Can not detected"
    orientation_error = "None"
    

    #Detect Can Presence and measure detection time
    pres_start_time = time.time()
    presence_result = presence_engine.ClassifyWithImage(
        img, threshold=0.55, top_k=1)
    pres_end_time = time.time()
    pres_inference_time = pres_end_time - pres_start_time
    print("Presence", presence_result)

    if not presence_result:
        previous_status = 1

    for result in presence_result:
        presence_prediction = presence_labels[result[0]]

        # Counter
        if((previous_status > result[0]) and ((time.time()-last_detection_time)>2)):
            print('New can detected')
            cnt = cnt+1
        previous_status = result[0]

        if(presence_prediction == 'Can detected'):
            #Record timestamp for last can detection
            last_detection_time=time.time()

            #Detect Can's orientation
            for result2 in orientation_engine.ClassifyWithImage(img, threshold=0.55, top_k=1):
                #result2= orientation_engine.ClassifyWithImage(img, threshold = 0.55, top_k=1)
                
                #Measure time taken for prediction
                #pres_end_time = time.time()
                #pres_inference_time = pres_end_time - pres_start_time

                print(result2)
                #Write the reults to CSV file as ouput which then will be sent to IoT core as MQTT payload
                data = (st,mac,result[0],result2[0],result2[1],cnt)
                

                #Write to json output
                #jsondata =  ({'infer_time':st,'device_id':mac,'can_presence':str(result[0]),'can_detect':str(result2[0]),'Orientation':str(result2[1]),'Counter':str(cnt)})
                with open('data.json','a', encoding="utf-8", newline='\r\n') as outfile:
                    json.dump({'infer_time':st,'device_id':str(mac),'can_presence':str(result[0]),'Orientation':str(result2[0]),'score':str(result2[1]),'Counter':str(cnt)}, outfile)
                    outfile.write('\n')
                    sys.stdout.flush()
                        
                orientation_prediction = orientation_labels[result2[0]]
                for orientation_notify in orientation_prediction:
                    orientation_error = (orientation_prediction == 'horizontal')
                    #orientation_error1 = (orientation_error, 'Error: Can placed incorrectly')
            #score = result[2]
            #print ('Score : ', result[2])

    text = 'Presence: '+presence_prediction
    draw.rectangle(((0,0),(230,110)), fill='white', outline='black')
    draw.text((5, 5), text=text, font=font, fill='blue')

    fps.update()
    fps.stop()
    text = 'Orientation: '+orientation_prediction
    draw.text((5, 20), text=text, font=font, fill='blue')

    text = 'Can counter: '+str(cnt)
    draw.text((5, 35), text=text, font=font, fill='blue')

    fps.update()
    fps.stop()

    #text = 'Time per inference: '
    #draw.text((5, 55), text=text, font=font, fill='blue')

    text = 'Time/prediction (ms): '+str(round(pres_inference_time,3)*1000)
    draw.text((5, 55), text=text, font=font, fill='blue')

    text = 'Alert: '+str(orientation_error)
    draw.text((5,75), text=text, font=font, fill='blue')

    
    # Display the resulting frame
    cv2.imshow('Video', numpy.array(img))

    fps.update()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cv2.destroyAllWindows()
stream.stop()
