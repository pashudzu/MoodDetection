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

EMOJI_NAME_MAP = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "neutral",
    5: "sad",
    6: "surprise",
}
EMOJI_IMAGE_MAP = {
    0: cv.imread(os.path.join("emojis", "angry_emoji.png"), cv.IMREAD_UNCHANGED),
    1: cv.imread(os.path.join("emojis", "disgust_emoji.png"), cv.IMREAD_UNCHANGED),
    2: cv.imread(os.path.join("emojis", "fear_emoji.png"), cv.IMREAD_UNCHANGED),
    3: cv.imread(os.path.join("emojis", "happy_emoji.png"), cv.IMREAD_UNCHANGED),
    4: cv.imread(os.path.join("emojis", "neutral_emoji.png"), cv.IMREAD_UNCHANGED),
    5: cv.imread(os.path.join("emojis", "sad_emoji.png"), cv.IMREAD_UNCHANGED),
    6: cv.imread(os.path.join("emojis", "surprise_emoji.png"), cv.IMREAD_UNCHANGED),
}

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
def get_emoji_img_by_id(id):
    return EMOJI_IMAGE_MAP.get(id)
def get_emoji_name_by_id(id):
    return EMOJI_NAME_MAP.get(id)

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
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Dropout(0.6),
            nn.Linear(64 * 6 * 6, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, 7),
        )
        self.loss_fn = nn.CrossEntropyLoss()
        self.optim = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-3)

    def forward(self, x):
        return self.model(x)

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
        return test_loss
    def detect_camera_emoji(self, face):
        torch.no_grad()
        self.model.eval() 
        pred = self.model(face).argmax(1)
        return pred

class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_model_state = None

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict()
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = model.state_dict()
            self.counter = 0
    def load_best_model(self, model):
        model.load_stat_dict(self.best_model_state)

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

early_stopper = EarlyStopping(patience=5, delta=0.001)

if not os.path.exists(saved_model_path):
    print(model)
    epochs = 20
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        model.train(train_dataloader)
        val_los = model.test(test_dataloader)
        print("Done!")

        early_stopper(val_los, model)
        if early_stopper.early_stop:
            print("🛑 Early stopping triggered.")
            break
    torch.save(model.state_dict(), saved_model_path)
else:
    model.load_state_dict(torch.load(saved_model_path, weights_only=True))
    print("Saved model loaded")

current_emoji_img = None

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
        face_transformed = transform(face_gray).to(device).unsqueeze(0)
        emoji_idx = float(model.detect_camera_emoji(face_transformed))
        emoji_img = np.array(get_emoji_img_by_id(emoji_idx))
        h_emoji, w_emoji = emoji_img.shape[:2]
        x_offset, y_offset = 10, 10
        if emoji_img.shape[2] == 4:
            alpha_channel = emoji_img[:, :, 3] / 255.0
            emoji_bgr = emoji_img[:, :, :3]
            roi = frame[y_offset:y_offset + h_emoji, x_offset:x_offset+w_emoji]
        else:
            alpha_channel = None
            emoji_brg = emoji_img
            roi[:] = emoji_brg
        if alpha_channel is not None:
            for c in range(3):
                roi[:, :, c] = roi[:, :, c] * (1 - alpha_channel) + emoji_bgr[:, :, c] * alpha_channel
        else:
            roi[:] = logo_bgr
        frame[y_offset:y_offset+h_emoji, x_offset:x_offset+w_emoji] = roi
        print(emoji_idx)

    cv.imshow("MoodDetection", frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        print("Exit key just pressed")
        break
cam.release()
cv.destroyAllWindows()