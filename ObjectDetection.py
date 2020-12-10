import os
import numpy as np
import cv2
from collections import Counter
import math
import time

labels = open('coco.names').read().strip().split('\n')
weights_path = 'yolov3-320.weights'
configuration_path = 'yolov3-320.cfg'

# Setting minimum probability to eliminate weak predictions
probability_minimum = 0.5

# Setting threshold for non maximum suppression
threshold = 0.3

network = cv2.dnn.readNetFromDarknet(configuration_path, weights_path)
layers_names_output = ['yolo_82', 'yolo_94', 'yolo_106']
network.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
network.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

def midpoint(p1, p2):
    return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)

def face_detection(frame):

    h, w = frame.shape[:2]
    flag = 0
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (320, 320), swapRB=True, crop=False)
    network.setInput(blob)  # setting blob as input to the network
    output_from_network = network.forward(layers_names_output)

    # Preparing lists for detected bounding boxes, obtained confidences and class's number
    bounding_boxes = []
    confidences = []
    class_numbers = []

    for result in output_from_network:
        # Going through all detections from current output layer
        for detection in result:
            # Getting class for current object
            scores = detection[5:]
            class_current = np.argmax(scores)

            # Getting confidence (probability) for current object
            confidence_current = scores[class_current]

            # Eliminating weak predictions by minimum probability
            if confidence_current > probability_minimum:
                # Scaling bounding box coordinates to the initial image size
                # YOLO data format keeps center of detected box and its width and height
                # That is why we can just element wise multiply them to the width and height of the image
                box_current = detection[0:4] * np.array([w, h, w, h])

                # From current box with YOLO format getting top left corner coordinates
                # that are x_min and y_min
                x_center, y_center, box_width, box_height = box_current.astype('int')
                x_min = int(x_center - (box_width / 2))
                y_min = int(y_center - (box_height / 2))

                # Adding results into prepared lists
                bounding_boxes.append([x_min, y_min, int(box_width), int(box_height)])
                confidences.append(float(confidence_current))
                class_numbers.append(class_current)

    results = cv2.dnn.NMSBoxes(bounding_boxes, confidences, probability_minimum, threshold)
    # Showing labels of the detected objects
    classFound = []
    for i in range(len(class_numbers)):
        classFound.append(labels[int(class_numbers[i])])

    classes = []
    final_boxes = [bounding_boxes[i] for i in results.flatten()]
    # if len(results) > 0:
    #     # Going through indexes of results
    #     for i in results.flatten():
    #         if classFound[i] == 'person' or classFound[i] == 'cell phone' or classFound[i] == 'laptop':
    #             classes.append(classFound[i])
    #             # Getting current bounding box coordinates
    #             x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
    #             box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]

    #             # Drawing bounding box on the original image
    #             cv2.rectangle(frame, (x_min, y_min), (x_min + box_width, y_min + box_height),
    #                         (255, 0, 0), 1)

    #             # Preparing text with label and confidence for current bounding box
    #             text_box_current = '{}'.format(classFound[i])

    #             # Putting text with label and confidence on the original image
    #             cv2.putText(frame, text_box_current, (x_min, y_min + 15), cv2.FONT_HERSHEY_DUPLEX,
    #                         0.7, (0, 0, 255), 2)
    cnt = Counter(classes)

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    info = {'objects_found': classes, 'bounding_boxes': final_boxes, 'num_of_objects': cnt}

    return info

# if __name__ == "__main__":
#     img = cv2.imread('my_pic.jpg')
#     info = face_detection(img)
#     print(info['objects_found'])
#     print(info['bounding_boxes'])
#     print(info['num_of_objects'])
