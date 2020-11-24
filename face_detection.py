import cv2
import numpy as np
import time

modelFile = "res10_300x300_ssd_iter_140000.caffemodel"
configFile = "deploy.prototxt.txt"

net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

#cap = cv2.VideoCapture(0)
#frame=cv2.imread('2020-10-30-201038.jpg')
def main(frame):
    
    frame = cv2.flip(frame, 1)

    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,(300, 300), (104.0, 117.0, 123.0))
    net.setInput(blob)
    faces = net.forward()
    #to draw faces on image
    face_count = 0
    face=[]
    for i in range(faces.shape[2]):
            confidence = faces[0, 0, i, 2]
            if confidence > 0.85:
                face_count = face_count + 1
                box = faces[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x1, y1) = box.astype("int")
                #cv2.rectangle(frame, (x, y), (x1, y1), (0, 0, 255), 2)
                face.append([x,y,x1,y1])

    # uncomment to show face_count in image
    #frame = cv2.putText(frame, f"No. of faces - {face_count}", (10,30), cv2.FONT_HERSHEY_COMPLEX, 1 , (0,0,0))
    #cv2.imshow('feed', frame)
    info={'faces':face,'face_count' : face_count}
    return info

#print(main(frame))
    # uncomment to print fps
    #print("FPS : ", round(1.0/ (time.time() - start), 2))

    

