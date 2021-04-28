import cv2
import numpy as np
from imgcat import imgcat
class FaceDetector(object):
    def __init__(self, model_path="./models/haarcascade_face.xml"):
        self.face_detector =cv2.CascadeClassifier('haarcascade_face.xml')
        self.face_detector.load('./models/haarcascade_face.xml')

    def detect_face(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(gray, 1.3, 5)
        domain_face = faces[0]
        return domain_face
    def detect_faces(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = srlf.face_detector.detectMultiScale(gray, 1.3, 5)
        return faces

    def crop_resize(self, img, face):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        x1, y1, w, h = face
        x2 = x1 + w
        y2 = y1 + h
        x1 += int(0.1 * w)
        x2 -= int(0.1 * w)
        y1 += int(0.1 * h)
        y2 -= int(0.1 * h)
        roi_gray = gray[y1:y2, x1:x2]
        roi_gray = cv2.resize(roi_gray, (256,256), interpolation=cv2.INTER_LINEAR)
        return roi_gray

    def draw_bbox(self, img, face):
        x1, y1, w, h = face
        x2 = x1 + w
        y2 = y1 + h
        draw_img = cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 1)
        return draw_img


