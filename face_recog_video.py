# PyImageSearch course practice on opencv
# Code originally by Adrian Rosebrock

from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2

# Parse commandline arguments, image, prototxt (,confidence) and model 
ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True, help="Path to input image")
ap.add_argument("-p", "--prototxt", required=True, help="Path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True, help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections") # default is 50%
args = vars(ap.parse_args())

print("[INFO] Loading model...")

# Load serialised model from disk
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# Initialise video stream and allow camera to warm up
url = ""
while url.strip() == "": url = input("Enter the video stream url > ")
stream = VideoStream(src=url).start()
time.sleep(2.0)


while True:
    # resize the frame to width of 400 px
    frame = stream.read()
    frame = imutils.resize(frame, width=400)

    # get frame dimensions, convert to 300x300 blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300,300)), 1.0, (300,300), (104.0,177.0,123.0) )

    # Pass the blob through the net and get detections
    net.setInput(blob)
    detections = net.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # filter out low confidence matches
        if confidence < args["confidence"]:
            continue

        # compute coordinates for bounding boxes
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        # write confidence and draw bounding boxes
        text = "{:.2f}%".format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        # cv2.rectangle(source, (startCoordinatesTuplePair), (endCoordinatesTuplePair), (colourTuple), thickness)
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
        cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,0),2)

    cv2.imshow("Frame", frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"): break

cv2.destroyAllWindows()
stream.stop()
    
