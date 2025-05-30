import cv2 as cv
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.accelerator as accelerator
import torch.optim as optim
import pandas as pd
import numpy as np
import zipfile
import os
from PIL import Image 
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image

face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

datasets_path = "datasets"
zip_dataset_file_path = os.path.join(datasets_path, "fer2013.zip")
zip_csv_file_path = os.path.join(datasets_path, "facialexpressionrecognition.zip")
test_dataset_path = os.path.join(datasets_path, "test")
train_dataset_path = os.path.join(datasets_path, "train")
scv_file_path = os.path.join(datasets_path, "fer2013.csv")
saved_model_path = os.path.join("model", "saved_model.pth")

cam = cv.VideoCapture(0)

device = accelerator.current_accelerator if accelerator.is_available() else "cpu"

print(f"device is {device}")

if not cam.isOpened():
    print("Камера не открыта!")
    exit()

def is_dataset_downloaded():
    if os.path.exists(zip_csv_file_path):
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
    def __init__(self, annotations_file, transform, train, data_path):
        self.transform = transform
        if (train):
            self.usage = 'Training'
        else:
            self.usage = 'PrivateTest'
        self.img_labels = pd.read_csv(annotations_file)
        self.usage_labels = self.img_labels[self.img_labels['Usage'] == self.usage]
        self.data_path = data_path
        print(self.usage_labels.head())
    def __len__(self):
        return len(self.usage_labels)
    def __getitem__(self, idx):
        pixels_string = self.usage_labels.iloc[idx, 1]
        pixels = np.fromstring(pixels_string, sep=" ", dtype=np.uint8).reshape((48, 48))
        image = Image.fromarray(pixels)
        
        if self.transform:
            image = self.transform(image)

        label = self.usage_labels.iloc[idx, 0]
        return image, label

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 512),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.5),
            nn.Linear(512, 7),
        )
        self.loss_fn = nn.CrossEntropyLoss()
        self.optim = optim.Adam(self.model.parameters(), 0.002)

    def forward(self, x):
        self.model(x)
        return(x)

    def train(self, data_loader):
        size = len(data_loader.dataset)
        self.model.train()
        for batch, (X, y) in enumerate(data_loader):
            X, y = X.to(device), y.to(device)
            pred = self.model(X)
            loss = self.loss_fn(pred, y)
            loss.backward()
            self.optim.step()
            self.optim.zero_grad()
            
            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    def test(self, data_loader):
        size = len(data_loader.dataset)
        test_loss, correct = 0, 0
        num_baches = len(data_loader)
        with torch.no_grad():
            for X, y in data_loader:
                X, y = X.to(device), y.to(device)
                pred = self.model(X)
                test_loss += self.loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_baches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

if not is_dataset_downloaded():
    os.system("kaggle datasets download -d nicolejyt/facialexpressionrecognition -p ./datasets")
    print("Dataset was downloaded")
else:
    print("Dataset has already downloaded")

if not is_dataset_unpacked():
    with zipfile.ZipFile(zip_csv_file_path, 'r') as zip_ref:
        zip_ref.extractall(datasets_path)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = FERDataset(zip_csv_file_path, transform, True, datasets_path)
test_dataset = FERDataset(zip_csv_file_path, transform, False, datasets_path)

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True, num_workers=0)

model = NeuralNetwork().to(device)

if not os.path.exists(saved_model_path):
    epochs = 10
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        model.train(train_dataloader)
        model.test(test_dataloader)
        print("Done!")
    torch.save(model.state_dict(), saved_model_path)
else:
    model.load_state_dict(torch.load(saved_model_path, weights_only=True))
    print("Saved model loaded")

while True:
    result, frame = cam.read()
    if not result:
        print("Here no result")
        break
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        face = frame[y:y+h, x:x+w]
        face_resized = cv.resize(face, (48, 48))
        face_gray = cv.cvtColor(face_resized, cv.COLOR_BGR2GRAY)
        face_transformed = transform(face_gray)
        
        print(face_transformed.shape)
        print(face_transformed.min(), face_transformed.max())

        face_transformed = face_transformed.to(device)
        face_4d = face_transformed.unsqueeze(0)
        cv.imshow("VideoCapture", face_gray)
        pred = model(face_4d).argmax(1)
        print(pred)

    #cv.imshow("VideoCapture", frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        print("Exit key just pressed")
        break
cam.release()
cv.destroyAllWindows()