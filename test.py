import cv2

from panda_producer import gen_panda_head
front_img = cv2.imread("face.jpg", 0)

res = gen_panda_head(front_img, "test", 128)

cv2.imwrite("res.jpg", res)
