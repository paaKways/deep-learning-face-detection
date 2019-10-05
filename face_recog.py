# PyImageSearch course practice on opencv
import numpy as np
import argparse
import cv2

# Parse commandline arguments, image, prototxt (,confidence) and model 
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to input image")
ap.add_argument("-p", "--prototxt", required=True, help="Path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True, help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections") # default is 50%
args = vars(ap.parse_args())

print("[INFO] Loading model...")

# Load serialised model from disk
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# Load input image
img = cv2.imread(args["image"])
(h, w) = img.shape[:2]

# Resize image to 300x300 pixels and normalise it to make a blob
blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0,177.0, 123.0))

# Pass the blob throught the network and obtain detections and predictions
print("[INFO] Computing object detections...")
net.setInput(blob)
detections = net.forward()

for i in range(0, detections.shape[2]):
    confidence = detections[0, 0, i, 2]

    if confidence > args["confidence"]:

        # Compute the (x,y)-coordinates of the bounding boxes for the object
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        text = "{:.2f}%".format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(img, (startX, startY), (endX, endY), (0,255,0), 2)
        cv2.putText(img, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,0), 2)

cv2.imshow('Output', img)
cv2.waitKey(0)
        
