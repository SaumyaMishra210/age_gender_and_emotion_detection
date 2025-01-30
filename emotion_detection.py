"""
this code checks for emotion in Input iamge
"""


import cv2
import numpy as np
from keras.models import load_model

model = load_model(r"Gender-and-Age-Detection\emotion_detection_model\complete_model_emotion_detector.h5")

# Load and preprocess the image
image_path = r'Gender-and-Age-Detection\girl1.jpg'
image = cv2.imread(image_path)
image = cv2.resize(image, (48, 48)) # Resize to match model input shape
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Convert to grayscale
image = np.reshape(image, [1, image.shape[0], image.shape[1], 1]) # Reshape to a 4D tensor

# Now, predict using the preprocessed image
predicted_class = np.argmax(model.predict(image)) 

emotion_dict= {'Angry': 0, 'Sad': 5, 'Neutral': 4, 'Disgust': 1, 'Surprise': 6, 'Fear': 2, 'Happy': 3}
label_map = dict((v,k) for k,v in emotion_dict.items()) 
predicted_label = label_map[predicted_class]
print(predicted_label)