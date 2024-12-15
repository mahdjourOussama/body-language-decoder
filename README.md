# README - Real-Time Emotion Detection using MediaPipe and OpenCV

## 📘 **Project Description**
This project aims to **detect human emotions in real-time via webcam** using the **MediaPipe** and **OpenCV** libraries. The system captures body landmarks to create a dataset and employs machine learning models to classify emotions into five main categories:

- 😊 **Happy**
- 😢 **Sad**
- ✌️ **Peace**
- 🤝 **Connecting**
- 👐 **Wakanda Forever**

This project was a small hobby project and an introduction to machine learning, focusing on hands-on experience in AI model implementation.

---

## 📐 **Project Structure**

```
├── body-language-decoder.py   # Module to extract body language landmarks using MediaPipe
├── body-language.pkl          # Best-performing trained machine learning model (Pickle file)
├── cords.csv                  # Custom training dataset of landmark coordinates and emotion labels
├── decoder.ipynb              # Jupyter notebook for training and evaluating the machine learning models
├── README.md                  # Project documentation (this file)
```

---

## ⚙️ **Key Features**

- **Real-time Emotion Detection**: Uses webcam input to detect human emotions live.
- **Custom Dataset Creation**: Uses MediaPipe to capture body landmarks and build a dataset for five emotional categories.
- **Machine Learning Models**: Trains and evaluates models like RandomForest, RidgeClassifier, GradientBoost, and LogisticRegression.
- **Model Deployment**: The best-performing model is saved as **body-language.pkl** for real-time inference.

---

## 🧪 **Models and Performance**

The following models were tested to classify emotions:

| **Model**                   | **Purpose**                | **Reason for Selection**                      |
|----------------------------|----------------------------|-----------------------------------------------|
| RandomForestClassifier      | Multi-class classification  | Handles high-dimensional feature spaces       |
| RidgeClassifier             | Regression-based classifier| Effective for linear relationships            |
| GradientBoostClassifier     | Boosted decision trees     | Improves performance via iterative learning  |
| LogisticRegression          | Binary and multi-class classifier | Interpretable and fast                      |

The **body-language.pkl** file contains the best-performing model, which is used for real-time inference.

---

## 🚀 **Installation and Usage**

### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/username/realtime-emotion-detection.git
cd realtime-emotion-detection
```

### **2️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
```

### **3️⃣ Run Real-Time Emotion Detection**
```bash
python body-language-decoder.py
```
This will activate your webcam and display a live emotion prediction based on detected body landmarks.

---

## 📦 **Data Collection Process**

The dataset is stored in **cords.csv**, which was created using the following process:
1. **Webcam Capture**: Used a webcam to record and capture a range of emotions.
2. **Landmark Detection**: MediaPipe extracts body landmarks from video frames.
3. **Data Labeling**: Each frame is labeled with one of the five target emotions (Happy, Sad, Peace, Connecting, Wakanda Forever).
4. **Dataset Storage**: Landmarks and labels are stored in **cords.csv** as structured data for training machine learning models.

---

## 🧪 **Training the Models**

### **1️⃣ Data Preparation**
- The dataset from **cords.csv** is loaded and preprocessed.
- The data is split into **training (80%)** and **testing (20%)** sets.

### **2️⃣ Model Training**
- The Jupyter notebook **decoder.ipynb** is used to train several models.
- Cross-validation is applied to avoid overfitting.

### **3️⃣ Model Saving**
- The best model is saved as **body-language.pkl** to be used for real-time prediction.

---

## 🧪 **Real-Time Emotion Prediction**

To **predict emotions in real-time**, follow these steps:
1. Run **body-language-decoder.py** to activate the webcam.
2. The system will predict and display the emotion of the person in front of the camera in real time.

---

## 🔥 **Challenges and Solutions**

| **Challenge**               | **Solution**                                      |
|----------------------------|--------------------------------------------------|
| Variability in expressions  | Captured a diverse range of expressions         |
| Real-time inference speed   | Saved best model with Pickle for fast loading     |
| Model overfitting           | Used cross-validation and a diverse dataset     |

---

## 📈 **Future Improvements**
- **Increase dataset size**: Capture more samples with diverse participants.
- **Model optimization**: Tune hyperparameters for better accuracy.
- **Enhance real-time performance**: Reduce latency for faster predictions.

---

## 🤝 **Contributors**
- **[Oussama Mahdjour]** - Project Owner & Developer


