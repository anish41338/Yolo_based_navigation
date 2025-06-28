
---

# 🦾 YOLO-Based Voice Navigation System

A Python project using **YOLOv5** object detection and speech output to assist visually impaired individuals with real-time voice navigation. The system processes video input, detects obstacles, and announces safe directions like “Turn Left”, “Move Forward”, etc.

---

## 🎯 Features

* ✅ **Real-time Object Detection** using YOLOv5
* 🎤 **Speech Output** using `pyttsx3` or `gTTS` for navigation instructions
* 📷 **Webcam Capture** for live video input
* 🧠 Logic to analyze path clearance and guide:

  * “Obstacle ahead, turn left/right”
  * “Clear path, move forward”

---

## 🛠️ Tech Stack

* **Python**
* **YOLOv5** (PyTorch-based object detection)
* **OpenCV** – video capture and frame processing
* **pyttsx3 / gTTS** – text-to-speech conversion
* **NumPy** – array processing and logic

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/anish41338/Yolo_based_navigation.git
cd Yolo_based_navigation
```

### 2. Install dependencies

```bash
pip install torch torchvision opencv-python pyttsx3 numpy
```

### 3. Run the system

```bash
python ULTIMATE.py
```

Or try:

```bash
python yolov5_detection.py
```

---

## 🧪 Files Description

| File                  | Purpose                                              |
| --------------------- | ---------------------------------------------------- |
| `yolov5_detection.py` | Runs object detection using YOLOv5                   |
| `web_cam_capture.py`  | Captures video feed via webcam                       |
| `navigate.py`         | Determines navigation decisions based on detection   |
| `ULTIMATE.py`         | Final integrated script (detection + voice guidance) |
| `README.md`           | This file                                            |
---

## 📬 Contact

Made with ❤️ by [Anish Sihag](mailto:anishsihag12@gmail.com)
GitHub: [@anish41338](https://github.com/anish41338)

---

