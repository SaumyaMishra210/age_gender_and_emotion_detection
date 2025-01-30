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
│── sample_images/               # Sample images
```

## How to Use
### 1. Upload the required models and images
Make sure the pre-trained models are available in the `/emotion_detection_model/` directory.

### 2. Run the `detect.py` script
Execute the script in **Google Colab** to upload an image and process it.

### 3. Upload an image
You will be prompted to upload an image for processing.

### 4. View the output
The script will detect the face, predict age, gender, and emotion, and display the image with annotations.

## Sample Images
Use the following links to download sample images for testing:
- [Sample Image 1](https://source.unsplash.com/300x300/?face)
- [Sample Image 2](https://source.unsplash.com/400x400/?human)
- [Sample Image 3](https://source.unsplash.com/500x500/?portrait)

## Output Example
After running the script, you will see results like:
```
Gender: Male
Age: 25-32 years
Emotion: Happy
```
The processed image will be saved in the `/outputs/` directory and displayed in the notebook.

## Author
Developed by **[Your Name]**

---
**Note:** This model's accuracy depends on lighting conditions, image quality, and model limitations.

