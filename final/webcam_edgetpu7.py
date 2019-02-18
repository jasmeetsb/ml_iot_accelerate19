import cv2
import os
import numpy
import time
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import imutils
from imutils.video import WebcamVideoStream

#Dependencies for IoT part
import sys
import json
import csv
import datetime

#Import edgeTPU Classification function
from edgetpu.classification.engine import ClassificationEngine
#Import edgeTPU Object Detection function
from edgetpu.detection.engine import DetectionEngine

# capture mac adress for identity of IoT device
from uuid import getnode as get_mac
mac = get_mac()


# Function to read labels from text files.
def ReadLabelFile(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    ret = {}
    for line in lines:
        pair = line.strip().split(':', maxsplit=1)
        ret[int(pair[0])] = pair[1].strip()
    return ret



print("Begin  creating engine instance")

## Orientation Detection - Specify the model and labels
orientation_model_path = './tflite_model/detect_coke_can_orientation/models_on-device_ICN6775484298763384503_jan20_edgetpu.tflite'
orientation_labels = ReadLabelFile(
    './tflite_model/detect_coke_can_orientation/orientation_labels.txt')

## Can Presence Detection - Specify the model and labels
presence_model_path = './tflite_model/detect_coke_can_presence/models_on-device_ICN2207431524157418399_jan20_edgetpu.tflite'
presence_labels = ReadLabelFile(
    './tflite_model/detect_coke_can_presence/presence_labels.txt')

# Can object localization - Specify the model and labels
obj_model_path = './tflite_model/obj_det_model/detect_1548129105537_edgetpu.tflite'
obj_labels = {1: 'coke_can'}

# Initialize the classification engines
orientation_engine = ClassificationEngine(orientation_model_path)
presence_engine = ClassificationEngine(presence_model_path)

##Initialize Object Detection engine
obj_engine = DetectionEngine(obj_model_path)
print("Classificatiopn and Detcetion engine instances created")


#Start VideoStream
stream = WebcamVideoStream(src=-1).start()
time.sleep(3.0)

# Draw Options - Select Font
font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 14)


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

    # Run object localization inference with edgetpu
    obj_det = obj_engine.DetectWithImage(img, threshold=0.05, relative_coord=False, top_k=1)


    if obj_det:
      for coke_can in obj_det:
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

        #Can Counter - To avoid double counting the same can, a delay of 2 seconds from last 
        #detection is added for the demo
        if((previous_status > result[0]) and ((time.time()-last_detection_time)>2)):
            print('New can detected')
            cnt = cnt+1
        previous_status = result[0]

        if(presence_prediction == 'Can detected'):
            #Record timestamp for last can detection
            last_detection_time=time.time()

            #Detect Can's orientation
            for result2 in orientation_engine.ClassifyWithImage(img, threshold=0.55, top_k=1):
 
                print(result2)
                #Write the results to CSV file as ouput which then will be sent to IoT core as MQTT payload
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


    #Display predicted information about the current frame
    text = 'Presence: '+presence_prediction
    draw.rectangle(((0,0),(230,110)), fill='white', outline='black')
    draw.text((5, 5), text=text, font=font, fill='blue')

    text = 'Orientation: '+orientation_prediction
    draw.text((5, 20), text=text, font=font, fill='blue')

    text = 'Can counter: '+str(cnt)
    draw.text((5, 35), text=text, font=font, fill='blue')

    text = 'Time/prediction (ms): '+str(round(pres_inference_time,3)*1000)
    draw.text((5, 55), text=text, font=font, fill='blue')

    text = 'Alert: '+str(orientation_error)
    draw.text((5,75), text=text, font=font, fill='blue')
    
    # Display the resulting frame
    cv2.imshow('Video', numpy.array(img))
    if cv2.waitKey(1) & 0xFF == ord('c'):
        break

# Cleanup
cv2.destroyAllWindows()
stream.stop()
