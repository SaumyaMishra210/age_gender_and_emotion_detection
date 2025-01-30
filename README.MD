# Age, Gender, and Emotion Detection

This project is a **face analysis tool** that detects a face in an image and predicts:
- **Age**
- **Gender**
- **Emotion**

## Features
- Uses **OpenCV's DNN module** for face detection.
- Implements **Deep Learning models** for age and gender classification.
- Uses **FER (Facial Expression Recognition)** for emotion detection.
- Compatible with **Google Colab**.

## Requirements
To run this project in **Google Colab**, install the required dependencies:
```bash
!pip install opencv-python numpy keras fer matplotlib
```

## Files and Directory Structure
```
/age_gender_and_emotion_detection
│── detect.py                   # Main script for detection
│── emotion_detection_model/     # Pre-trained models
│   ├── opencv_face_detector.pbtxt
│   ├── opencv_face_detector_uint8.pb
│   ├── age_deploy.prototxt
│   ├── age_net.caffemodel
│   ├── gender_deploy.prototxt
│   ├── gender_net.caffemodel
│── outputs/                     # Output images
│── images/                      # Sample images
```

## How to Use
### 1. Upload the required models and images
Make sure the pre-trained models are available in the `/models/` directory.

### 2. Run the `detect.py` script
Execute the script in **Google Colab** to upload an image and process it.

### 3. Upload an image
You will be prompted to upload an image for processing.

### 4. View the output
The script will detect the face, predict age, gender, and emotion, and display the image with annotations.

## Sample Images
![Alt Text](Gender-Emotion-and-Age-Detection/images/girl1.jpg)

## Output Example
After running the script, you will see results like:
```
Gender: Female
Age: 25-32 years
Emotion: Happy
```
![Alt Text](Gender-Emotion-and-Age-Detection\outputs\output.jpg)

The processed image will be saved in the `/outputs/` directory and displayed in the notebook.


---
**Note:** This model's accuracy depends on lighting conditions, image quality, and model limitations.

