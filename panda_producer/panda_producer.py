import cv2
import numpy as np

def merge_panda_facer(back_img, front_img, hyper=130):
    
    if len(front_img.shape) >=3:
        front_img = cv2.cvtColor(front_img,cv2.COLOR_RGB2GRAY)
    front_img[np.where(front_img > hyper)] = 255
    front_img = cv2.resize(front_img, (256, 256), interpolation=cv2.INTER_LINEAR)
    
    back_img = cv2.resize(back_img, (512, 512), interpolation=cv2.INTER_LINEAR)

    back_h, back_w = back_img.shape
    front_h, front_w = front_img.shape

    start_x = int((back_w - front_w) / 2)
    end_x = start_x + front_w

    start_y = int((back_h - front_h) / 2)
    start_y = int(0.6 * start_y)
    end_y = front_h + start_y
    back_img[start_y: end_y, start_x: end_x] = front_img

    return back_img

def add_text(back_img, text):
    cv2.putText(back_img, text, (130, 480), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 4)

def gen_panda_head(front_img, text=None, hyper=128):
    back_img = np.load("panda.npy")
    merge_img = merge_panda_face(back_img,front_img, int(hyper))
    if text is not None:
        add_text(merge_img, str(text))
    return merge_img

