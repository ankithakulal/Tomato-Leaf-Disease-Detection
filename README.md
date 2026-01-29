# Tomato-Leaf-Disease-Detection

This project detects tomato leaf diseases using a deep learning CNN model
and provides visual explanations using Grad-CAM along with voice output.

## Technologies Used
- Python
- Flask
- TensorFlow / Keras
- OpenCV
- CNN
- Grad-CAM
- gTTS (Text to Speech)

## Project Workflow
1. User uploads a tomato leaf image
2. Image is preprocessed and passed to CNN model
3. Disease is predicted with confidence score
4. Grad-CAM heatmap is generated
5. Audio explanation is generated for results

## Output
- Disease name
- Confidence percentage
- Heatmap visualization
- Voice explanation

## Note
Due to large model size and system limitations, the trained `.h5` model file
and dataset are not included in this repository.
This repository focuses on application logic and workflow.
