""" ui looks good enough 
    Need to connect it with google colab /google Drive/Api links between them
 """



import gradio as gr
import random
import cv2
import numpy as np
from PIL import Image

# Sample image path (Replace with an actual image if needed)
SAMPLE_IMAGE_PATH = r"outputs\output (2).jpg"

# Sample data for UI
SAMPLE_EMOTIONS = ["Happy", "Sad", "Angry", "Neutral", "Surprised"]
SAMPLE_GENDERS = ["Male", "Female"]
SAMPLE_AGE_GROUPS = ["0-2", "4-6", "8-12", "15-20", "25-32", "38-43", "48-53", "60-100"]

# Function to generate sample results (simulated ML output)
def predict_emotion(image):
    if image is None:
        image = Image.open(SAMPLE_IMAGE_PATH)  # Load sample image if none is uploaded
    
    # Convert PIL Image to OpenCV format
    image_cv = np.array(image)
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)

    # Simulate ML predictions using random choices
    detected_emotion = "Happy"
    detected_gender = "Female"
    detected_age = "15-20"

    # Add text overlay for visualization
    label = f"{detected_gender}, {detected_age}, {detected_emotion}"
    cv2.putText(image_cv, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Convert back to PIL Image for Gradio
    processed_image = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))

    # Return both the processed image and predictions
    return processed_image, {
        "Emotion": detected_emotion,
        "Gender": detected_gender,
        "Age Group": detected_age
    }

# UI Layout using Gradio Blocks
with gr.Blocks(theme="soft") as demo:
    gr.Markdown("# 👨‍💻 Emotion, Age, and Gender Detector")
    gr.Markdown("Upload an image, and the system will predict emotions, gender, and age. (Using sample data due to lack of GPU support)")

    with gr.Row():
        image_input = gr.Image(label="Upload Image", type="pil", interactive=True)
        output_image = gr.Image(label="Processed Image")
        output_data = gr.JSON(label="Predictions")

    with gr.Row():
        submit_button = gr.Button("Detect")

    submit_button.click(predict_emotion, inputs=image_input, outputs=[output_image, output_data])

# Run the Gradio app
if __name__ == "__main__":
    demo.launch(share=True)
