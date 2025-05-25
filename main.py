import cv2

cam = VideoCaputure(0)
result, image = cam.read()

if result:
    imshow("cam", image)
    imwrite("camING.phg", image)