import os
import numpy as np
import cv2
import dlib
from collections import Counter
import math
import time

detector = dlib.get_frontal_face_detector()
predictor2 = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

lbls = ['Number of Faces: ', 'Blinking: ', 'Looking at the Screen: ']

def midpoint(p1, p2):
    return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)


def get_blinking_ratio(eye_points, facial_landmarks):
    left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))
    hor_line_length = math.hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    ver_line_length = math.hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))
    ratio = hor_line_length / ver_line_length
    return ratio

def object_detection(frame):

    h, w, c = frame.shape
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    no_faces = len(faces)
    blinking = False
    looking_screen = True
    t = [faces, blinking, looking_screen]
    t[0] = no_faces
    if no_faces == 0:
        looking_screen = False
        t[2] = False

    for face in faces:
        landmarks = predictor2(gray, face)
        left_eye_ratio = get_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks)
        right_eye_ratio = get_blinking_ratio([42, 43, 44, 45, 46, 47], landmarks)
        blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2
        if blinking_ratio > 4.3:
            blinking = True
            t[1] = True

    # offset = 0
    # for itr, word in enumerate(lbls):
    #     offset += int(h / len(lbls)) - 10
    #     cv2.putText(frame, word + str(t[itr]), (20, offset), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 1)
    
    info = {'faces' : faces, 'num_faces': len(faces),'blinking': blinking, 'looking_at_screen': looking_screen}
    return info


# if __name__ == "__main__":
#     img = cv2.imread('2faces.jpg')
#     info = object_detection(img)
#     print(info['num_faces'])
