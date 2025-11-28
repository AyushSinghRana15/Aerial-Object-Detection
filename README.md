# Aerial Object Decision 🛰️🕊️

A deep learning project for **aerial image classification**, distinguishing between **birds** and **drones** using transfer learning (ResNet50, MobileNetV2, EfficientNetB0) and a custom CNN. The best model is deployed as an interactive **Streamlit web app** with support for both image upload and live camera capture.

## Features

- 🔍 Classifies aerial images into **Bird** or **Drone**
- 🧠 Multiple models:
  - Custom CNN
  - ResNet50 (best performing)
  - MobileNetV2
  - EfficientNetB0
- 📊 Evaluation with accuracy, precision, recall, F1-score, and confusion matrices
- 🌐 Streamlit app:
  - Image upload
  - Live camera capture
  - Model selection and confidence scores
- 💾 Large model weights managed via **Git LFS**

## Tech Stack

- Python, PyTorch, Torchvision
- Streamlit
- Scikit-learn
- Git & Git LFS

## Project Structure

├── aerial.py                  # Streamlit app 

├── Aerial Classification.ipynb  # Training & analysis notebook 

├── classification_Dataset/    # Classification dataset 

├── object_detection_Dataset/  # detection dataset

├── *.pth / *.pt               # Trained model weights (Git LFS)

├── requirements.txt           # Python dependencies 

└── runtime.txt                # Python version for deployment


## How to Run Locally

git clone https://github.com/AyushSinghRana15/Aerial-Object-Decision.git

cd Aerial-Object-Decision

pip install -r requirements.txt 

streamlit run aerial.py

Then open the URL shown in the terminal (usually `http://localhost:8501`) to use the app.

## Results

- **ResNet50** achieved the best performance (~98% accuracy) on the bird vs drone classification task.
- Lighter models (MobileNetV2, EfficientNetB0) and the custom CNN are included for comparison and experimentation.

## Future Work

- Extend to multi-class aerial objects
- Use a larger dataset with lower quality images
- Optimize lightweight models for edge deployment
- Add uncertainty estimation and better calibration

## Trial Vedio


https://github.com/user-attachments/assets/27e4310f-48a6-420f-b0f3-05517778e4a9

