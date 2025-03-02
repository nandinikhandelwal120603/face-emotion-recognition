# AI Face and Emotion Recognition

## 📌 Project Overview
This project performs **face recognition** and **emotion classification** using OpenCV, TensorFlow/Keras, and a trained deep learning model. It can detect faces, recognize users, and classify emotions in real-time.

---

## 📂 Project Structure

```
C:\Users\khand\OneDrive\Documents\python\ai face and emotion recognition
📄 app.py
📄 emotion.py
📂 face/
  📄 haarcascade_frontalface_default.xml
  📂 media/
  📄 README.md
  📄 requirements.txt
  📂 src/
    📄 combined.py
    📄 face_recognizer.py
    📄 face_taker.py
    📄 face_trainer.py
    📄 just_a_file.py
    📂 settings/
      📄 settings.py
      📄 __init__.py
      📂 __pycache__/
        📄 settings.cpython-312.pyc
        📄 __init__.cpython-312.pyc
    📄 __init__.py
📄 face.py
📄 haarcascade_frontalface_default.xml
📂 images/
📄 main.py
📄 model.h5
📄 multi.py
📄 names.json
📄 README.md
📄 real-time-facial-emotion-classification-cnn-using-keras.ipynb
📄 requirements.txt
📂 templates/
  📄 index.html
📄 trainer.yml
```

---

## 🚀 Features
- **Face Detection**: Uses OpenCV's `haarcascade_frontalface_default.xml`.
- **Face Recognition**: Identifies and recognizes trained users.
- **Emotion Classification**: Uses a trained CNN model (`model.h5`) to detect emotions like happy, sad, angry, etc.
- **Real-time Tracking**: Captures and tracks faces dynamically.
- **Data Management**: Allows users to add, train, and delete their facial data.

---

## 🔧 Installation & Setup

### 1️⃣ **Clone the Repository**
```sh
git clone https://github.com/your-username/your-repo-name.git
cd ai-face-and-emotion-recognition
```

### 2️⃣ **Create & Activate Virtual Environment**
```sh
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate    # On Windows
```

### 3️⃣ **Install Dependencies**
```sh
pip install -r requirements.txt
```

### 4️⃣ **Run the Application**
```sh
python main.py
```

---

## 🛠️ Usage
1. **Train the Model**
   - Run `face_taker.py` to capture face images.
   - Execute `face_trainer.py` to train the recognition model.
2. **Real-time Recognition & Emotion Detection**
   - Run `face_recognizer.py` for face recognition.
   - Execute `emotion.py` for emotion classification.

---

## 📝 Configuration
Modify `settings/settings.py` to change parameters such as:
- Camera index
- Face detection sensitivity
- Paths for storing images and models

---

## 📜 License
This project is open-source and available under the **MIT License**.

---

## 👨‍💻 Author
- **Nandini kahndelwal* – [GitHub Profile](https://github.com/your-username)

---

## 🤝 Contributing
Feel free to fork, submit issues, or open pull requests to improve this project!
****
