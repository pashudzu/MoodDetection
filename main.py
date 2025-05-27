import cv2 as cv
import zipfile
import os

face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

datasets_path = "datasets"
zip_file_path = os.path.join(datasets_path, "fer2013.zip")
test_dataset_path = os.path.join(datasets_path, "test")
train_dataset_path = os.path.join(datasets_path, "train")

cam = cv.VideoCapture(0)

if not cam.isOpened():
    print("Камера не открыта!")
    exit()

def is_dataset_downloaded():
    if os.path.exists(zip_file_path):
        return True
    else:
        return False
def is_dataset_unpacked():
    if os.path.exists(test_dataset_path) and os.path.exists(train_dataset_path):
        print("Dataset has already unpacked")
        return True
    else:
        return False

if not is_dataset_downloaded():
    os.system("kaggle datasets download -d msambare/fer2013 -p ./datasets")
    print("Dataset was downloaded")
else:
    print("Dataset has already downloaded")

if not is_dataset_unpacked():
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        print("Unpacking dataset")
        zip_ref.extractall(datasets_path)

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