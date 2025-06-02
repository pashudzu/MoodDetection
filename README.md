# 🎭 Real-Time Facial Emotion Recognition (FER) using CNN, PyTorch & OpenCV
This project detects human emotions in real time using your webcam. It uses a Convolutional Neural Network (CNN) trained on the FER2013 dataset. Faces are detected using OpenCV with Haar cascades, and an appropriate emoji is overlaid on the image based on the recognized emotion.

## 📸 Screeshots
![screenshot](https://github.com/user-attachments/assets/8809747b-96bc-466b-8f51-9e0f0d4b3cba)

## 📦 Dependencies
Required libraries:

```
pip install torch torchvision opencv-python pandas numpy pillow
```
You'll also need the kaggle CLI to download the dataset:
```
pip install kaggle
```
Save your Kaggle API key in ~/.kaggle/kaggle.json or configure environment variables properly.

## 📁 Project Structure  
├── datasets/  
│   ├── fer2013.csv  
│   └── facialexpressionrecognition.zip  
├── emojis/  
│   ├── happy_emoji.png  
│   ├── sad_emoji.png  
│   └── ... (other emotions)  
├── model/  
│   └── saved_model.pth  
├── main.py  
├── README.md  


## 😃 Recognized Emotions
The model recognizes the following emotions:

😠 Angry
🤢 Disgust
😱 Fear
😄 Happy
😐 Neutral
😢 Sad
😲 Surprise

## 🚀 How to Run
Make sure the webcam works.

Run the script with python main.py

If the model is not found, training will start automatically.

The app will open a camera window showing detected faces with overlaid emojis.

## 🧠 Model Summary
A custom CNN trained on FER2013 with:

Convolutional + BatchNorm + LeakyReLU + MaxPooling layers

Dropout and Fully Connected layers

Trained using Adam optimizer and CrossEntropyLoss

## ⚠️ Troubleshooting
Check your webcam and OpenCV installation if the app doesn’t start.

Make sure you have the correct version of PyTorch for your system.

## ✍️ Author
Developed by a student ML developer.

## 📄 License
This project is licensed under the MIT License.
