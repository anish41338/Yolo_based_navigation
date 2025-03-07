import torch
import cv2
import numpy as np
from gtts import gTTS
import os
import pygame
import threading
import time


pygame.mixer.init()


def speak(text):
    try:
        timestamp = int(time.time() * 1000)  
        file_path = f"output_{timestamp}.mp3"
        tts = gTTS(text, lang='en')
        tts.save(file_path)
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.2)  
        os.remove(file_path)  
    except Exception as e:
        print(f"Error in speak function: {e}")

def audio_thread(text):
    threading.Thread(target=speak, args=(text,)).start()


print("Loading YOLOv5 model...")
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', force_reload=True)
print("Model loaded successfully!")


cap = cv2.VideoCapture(0)  

while True:
    ret, frame = cap.read()
    if not ret:
        break

    
    results = model(frame)
    labels, confidences, boxes = results.xyxyn[0][:, -1], results.xyxyn[0][:, -2], results.xyxyn[0][:, :-2]

    detected_objects = []

    
    for label, confidence, box in zip(labels, confidences, boxes):
        x1, y1, x2, y2 = int(box[0] * frame.shape[1]), int(box[1] * frame.shape[0]), int(box[2] * frame.shape[1]), int(box[3] * frame.shape[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{model.names[int(label)]} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        detected_objects.append(model.names[int(label)])

    if detected_objects:
        audio_thread("Detected: " + ", ".join(detected_objects))

    time.sleep(5)  

    cv2.imshow('YOLOv5 Object Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
