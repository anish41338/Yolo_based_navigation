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
            time.sleep(0.5)
        os.remove(file_path)
    except Exception as e:
        print(f"Error in speak function: {e}")

def audio_thread(text, delay=0):
    def delayed_speak(text, delay):
        time.sleep(delay)
        speak(text)
    threading.Thread(target=delayed_speak, args=(text, delay)).start()

print("Loading YOLOv5 model...")
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', force_reload=True)
print("Model loaded successfully!")

cap = cv2.VideoCapture(0)
prev_frame = None
last_command_time = time.time()
command_delay = 5  # Minimum delay between commands in seconds

# To track detected objects and avoid repetitive announcements
detected_objects_memory = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    labels, confidences, boxes = results.xyxyn[0][:, -1], results.xyxyn[0][:, -2], results.xyxyn[0][:, :-2]

    detected_objects = []
    objects_to_announce = []

    for label, confidence, box in zip(labels, confidences, boxes):
        x1, y1, x2, y2 = int(box[0] * frame.shape[1]), int(box[1] * frame.shape[0]), int(box[2] * frame.shape[1]), int(box[3] * frame.shape[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{model.names[int(label)]} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        detected_objects.append((model.names[int(label)], x1, y1, x2, y2))

        # Announce only if it's a new object or the object has moved significantly
        obj_key = (model.names[int(label)], x1, y1, x2, y2)
        if obj_key not in detected_objects_memory:
            objects_to_announce.append(model.names[int(label)])
            detected_objects_memory[obj_key] = time.time()

    # Clean up old entries from memory
    current_time = time.time()
    detected_objects_memory = {key: value for key, value in detected_objects_memory.items() if current_time - value < 10}

    if objects_to_announce:
        audio_thread("Detected: " + ", ".join(objects_to_announce))

    time.sleep(3)  # Delay between detection announcement and navigation command

    # Determine empty spaces with finer granularity
    frame_height, frame_width = frame.shape[:2]
    grid_size = 10  # Using a reasonable grid size for accuracy
    grid = np.zeros((grid_size, grid_size), dtype=int)

    for obj in detected_objects:
        x1, y1, x2, y2 = obj[1], obj[2], obj[3], obj[4]
        grid_x1, grid_y1 = x1 // (frame_width // grid_size), y1 // (frame_height // grid_size)
        grid_x2, grid_y2 = x2 // (frame_width // grid_size), y2 // (frame_height // grid_size)
        grid[grid_y1:grid_y2+1, grid_x1:grid_x2+1] = 1

    left_empty = np.sum(grid[:, :grid_size//3]) == 0
    center_empty = np.sum(grid[:, grid_size//3:2*grid_size//3]) == 0
    right_empty = np.sum(grid[:, 2*grid_size//3:]) == 0

    # Generate navigation command based on clear regions
    if current_time - last_command_time > command_delay:  # Ensure minimum delay between commands
        if left_empty and not center_empty and not right_empty:
            audio_thread("Turn left, empty space detected.", delay=4)
        elif right_empty and not center_empty and not left_empty:
            audio_thread("Turn right, empty space detected.", delay=4)
        elif center_empty:
            audio_thread("Move forward.", delay=4)
        else:
            audio_thread("No clear path detected.", delay=4)
        last_command_time = current_time

    cv2.imshow('YOLOv5 Object Detection and Navigation', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
