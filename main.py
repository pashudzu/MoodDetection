import cv2 as cv
import torch.nn as nn
import torchvision.transforms as transforms
import torch.accelerator as accelerator
import torch.optim as optim
import pandas as pd
import numpy as np
import zipfile
import os
from torch.utils.data import Dataset
from torchvision.io import read_image

face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

datasets_path = "datasets"
zip_dataset_file_path = os.path.join(datasets_path, "fer2013.zip")
zip_csv_file_path = os.path.join(datasets_path, "facialexpressionrecognition.zip")
test_dataset_path = os.path.join(datasets_path, "test")
train_dataset_path = os.path.join(datasets_path, "train")
scv_file_path = os.path.join(datasets_path, "fer2013.csv")

cam = cv.VideoCapture(0)

device = accelerator.current_accelerator if accelerator.is_available() else "cpu"

print(f"device is {device}")

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

class FERDataset(Dataset):
    def __init__(self, annotations_file, transform, train):
        self.transform = transform
        if (train):
            self.usage = 'Training'
        else:
            self.usage = 'PrivateTest'
        self.img_labels = pd.read_csv(annotations_file)
        self.usage_labels = self.img_labels[self.img_labels['Usage'] == self.usage]
    def __len__(self):
        return len(self.usage_labels)
    def __getitem__(self, idx):
        img_path = self.usage_labels.iloc[idx, 0]
        image = read_image(img_path)
        label = self.usage_labels.iloc[idx, 1]
        return image, label

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(64 * 12 * 12, 512),
            nn.LeakyReLU(0.01),
            nn.Linear(512, 7),
        )
        self.loss_fn = nn.CrossEntropyLoss()
        self.optim = optim.Adam(self.model.parameters(), 0.002)

    def forward(self, x):
        return self.model(x)

    #def train(self, epochs, batch_size, data_loader):
    #    for epoth in range(epothes):
    #        for batch, (X, y) in enumerate()

model = NeuralNetwork().to(device)

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
    with zipfile.ZipFile(zip_csv_file_path, 'r') as zip_ref:
        zip_ref.extractall(datasets_path)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = FERDataset(zip_csv_file_path, transform, True)
test_dataset = FERDataset(zip_csv_file_path, transform, False)

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