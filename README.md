# 🤟 LingoHands, an AI-Powered ASL Detection : Translating Gestures, Connecting Worlds 🌍🤝

## 🌍 Bridging the Communication Gap

LingoHands is an AI-powered real-time American Sign Language (ASL) detection system designed to enhance accessibility for the hearing-impaired. By leveraging deep learning and computer vision, it accurately recognizes ASL signs from images and video streams, fostering inclusivity in a communication-driven world.

---
## ✨ Features
✅ **ASL Alphabet Recognition** – Detects all 26 ASL letters plus special signs (space, delete, nothing).  
✅ **Deep Learning Model** – Powered by a CNN for precise sign language interpretation.  
✅ **Real-Time Prediction** – Recognizes ASL gestures instantly from a webcam feed.  
✅ **Preprocessed Dataset** – Utilizes a curated ASL dataset for robust training and validation.  
✅ **User-Friendly Interface** – Easily deployable for real-world applications.  

---
## 📂 Dataset Details
LingoHands is trained on the **Kaggle ASL Alphabet dataset**, which includes:
- **26 Classes**: Letters A-Z plus additional gestures.
- **Optimized Image Processing**: Images resized to 64x64 pixels for efficient model training.
- **Comprehensive Test Set**: Ensures accurate model evaluation.

The dataset was preprocessed using OpenCV and NumPy, including normalization and resizing, to ensure optimal performance during model training.

---
## 🧠 Model Architecture
LingoHands employs a **Convolutional Neural Network (CNN)** for image classification, comprising:
1️⃣ **Convolutional Layers** – Extracts essential features from input images.  
2️⃣ **MaxPooling Layers** – Reduces spatial dimensions and minimizes overfitting.  
3️⃣ **Fully Connected Layers** – Maps extracted features to ASL sign labels.  
4️⃣ **Softmax Activation** – Predicts the most probable ASL sign.  

The model was trained using TensorFlow and Keras, with categorical cross-entropy as the loss function and Adam optimizer for efficient learning.

---
## 🎯 Training & Results
🚀 **Training was conducted on Google Colab**, using GPU acceleration for faster computation. The training process involved:
- **Batch size of 32** and **10 epochs** for efficient learning.
- **80-20 train-validation split** to ensure generalization.
- **Accuracy of over 90%** achieved on validation data.

### 📈 Performance metrics include:
- Training and validation accuracy/loss trends.
- Real-time testing with new ASL sign images.

### 📊 Model Performance Visualization
Below is an example of accuracy and loss curves:

```python
import matplotlib.pyplot as plt

# Plot accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Model Accuracy')
plt.show()

# Plot loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Model Loss')
plt.show()
```
![image](https://github.com/user-attachments/assets/b7a35b04-f042-4d75-a671-0fe38b6d6ba8)

---
## 🌍 Real-World Applications
LingoHands is more than just an AI model; it’s a step toward accessibility and inclusion. Possible applications include:
- 🔹 **ASL Translation Apps** – Enables real-time communication between deaf and hearing individuals.
- 🔹 **Educational Tools** – Assists ASL learners with interactive practice.
- 🔹 **Smart Assistants** – Recognizes ASL gestures for seamless smart home integration.
- 🔹 **Healthcare & Emergency Services** – Helps patients communicate with medical professionals.

---
## 🎬 Demo
To test the model with a sample ASL image:
```python
from tensorflow.keras.models import load_model
import cv2
import numpy as np

# Load the trained model
model = load_model("LingoHands_CNN_Model.h5")

def predict_asl_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (64, 64))
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    prediction = model.predict(img)
    predicted_label = np.argmax(prediction)
    class_name = classes[predicted_label]

    print(f"Predicted Sign: {class_name}")
    return class_name

# Example usage
sample_img_path = "path/to/sample_image.jpg"
predict_asl_image(sample_img_path)
```
![image](https://github.com/user-attachments/assets/7a4cfe9a-d320-4312-bda0-5baa7c89017c)

---
## 🔮 Future Enhancements
🚀 Support for **dynamic ASL gestures** beyond alphabets.  
📲 Deployment as a **web & mobile app**.  
⚡ Optimization for **faster real-time processing**.  
🤖 Integration with **voice assistants** for seamless communication.  

---
