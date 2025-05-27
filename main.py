import cv2 as cv

face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

cam = cv.VideoCapture(0)

if not cam.isOpened():
    print("Камера не открыта!")
    exit()

while True:
    result, frame = cam.read()
    if not result:
        print("Here no result")
        break
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv.imshow("VideoCapture", frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        print("Exit key just pressed")
        break
cam.release()
cv.destroyAllWindows()