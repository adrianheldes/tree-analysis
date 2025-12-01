import cv2
import numpy as np
from tensorflow.keras.models import load_model
from video_reader import VideoReader
from tree

# Load pre-trained model for plant classification
model = load_model('path_to_your_model.h5')

def preprocess_frame(frame):
    # Resize frame to the input size of the model
    frame = cv2.resize(frame, (224, 224))
    frame = frame / 255.0  # Normalize the frame
    return np.expand_dims(frame, axis=0)

def classify_plant(frame):
    processed_frame = preprocess_frame(frame)
    predictions = model.predict(processed_frame)
    class_index = np.argmax(predictions, axis=1)
    return class_index

def analyze_video(video_path):
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        plant_class = classify_plant(frame)
        print(f'Plant class: {plant_class}')

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = 'path_to_your_video.mp4'
    analyze_video(video_path)