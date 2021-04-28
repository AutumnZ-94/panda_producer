import cv2, sys
import numpy as np
from panda_producer.panda_producer import gen_panda_head
from panda_producer import build_face_model
from imgcat import imgcat

def demo(img_path = "raw_img.jpg", mode='default', hyper=128):
    detector = build_face_model.FaceDetector()
    front_img = cv2.imread(img_path)
    face = detector.detect_face(front_img)
    crop_img = detector.crop_resize(front_img, face)
    if mode == 'default':
        hyper = crop_img.mean() 
    if mode == 'manual':
        hyper = crop_img.mean() 
    crop_img[np.where(crop_img > hyper)] = 255

    res = gen_panda_head(crop_img, "test", 128)
    cv2.imwrite("res_" + img_path.split('/')[-1] , res)

if __name__ == "__main__":
    img_path = sys.argv[1]
    demo(img_path)

