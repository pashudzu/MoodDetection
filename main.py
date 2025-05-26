import cv2 as cv

cam = cv.VideoCapture(0)

if not cam.isOpened():
    print("Камера не открыта!")
    exit()

while True:
    result, frame = cam.read()
    if not result:
        print("Here no result")
        break
    cv.imshow("VideoCapture", frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        print("Exit key just pressed")
        break
cam.release()
cv.destroyAllWindows()