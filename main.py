import cv2 as cv
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import zipfile
import os

face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

datasets_path = "datasets"
zip_dataset_file_path = os.path.join(datasets_path, "fer2013.zip")
zip_csv_file_path = os.path.join(datasets_path, "facialexpressionrecognition.zip")
test_dataset_path = os.path.join(datasets_path, "test")
train_dataset_path = os.path.join(datasets_path, "train")
scv_file_path = os.path.join(datasets_path, "fer2013.csv")

cam = cv.VideoCapture(0)

if not cam.isOpened():
    print("Камера не открыта!")
    exit()

def is_dataset_downloaded():
    if os.path.exists(zip_dataset_file_path) and os.path.exists(zip_csv_file_path):
        return True
    else:
        return False
def is_dataset_unpacked():
    if os.path.exists(test_dataset_path) and os.path.exists(train_dataset_path) and os.path.exists(scv_file_path):
        print("Dataset has already unpacked")
        return True
    else:
        return False

if not is_dataset_downloaded():
    os.system("kaggle datasets download -d msambare/fer2013 -p ./datasets")
    os.system("kaggle datasets download -d nicolejyt/facialexpressionrecognition -p ./datasets")
    print("Dataset was downloaded")
else:
    print("Dataset has already downloaded")

if not is_dataset_unpacked():
    with zipfile.ZipFile(zip_dataset_file_path, 'r') as zip_ref:
        print("Unpacking dataset")
        zip_ref.extractall(datasets_path)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

#class FERDataset(Dataset):


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNerwork, self).__init__()

        self.loss_fn = nn.CrossEntropyLoss()
        self.optim = torch.optim.Adam(self.generator.parameters(), 0.002)

        self.model = nn.Sequential(
            nn.Conv2d(1, 32, karnel_size=3, padding=1),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, karnel_size=3, padding=1),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(64 * 12 * 12, 512),
            nn.LeakyReLU(0.01),
            nn.Linear(512, 7),
        )
    def forward(self, x):
        return self.model(x)

    #def train(self, epochs, batch_size, data_loader):
    #    for epoth in range(epothes):


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