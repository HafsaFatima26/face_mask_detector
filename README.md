
# Face Mask Detection Using Deep Learning

This repository contains an end-to-end real-time face mask detection system using deep learning and computer vision. The project detects faces in webcam video and classifies each as “Mask” or “No Mask,” providing instant visual feedback. It is ideal for safety monitoring in public and workplace environments.

---

## Features

- Detects multiple faces in real-time video from your webcam
- Classifies each face as “Mask” or “No Mask”
- Lightweight, accurate, and works efficiently on standard hardware
- Visualizes results with colored bounding boxes and confidence labels
- Easy to retrain on your own images

---

## Technologies Used

- **Language:** Python 3
- **Libraries:** TensorFlow (Keras), OpenCV, imutils, scikit-learn, NumPy, Matplotlib
- **Model Architecture:** MobileNetV2 (transfer learning)
- **IDE:** Visual Studio Code

---

## How It Works

### 1. Model Training

- Collect and organize images of faces with and without masks
- Images are resized and preprocessed
- MobileNetV2 (pretrained) is used as a feature extractor
- A custom classification head is trained for the mask/no-mask task
- Data augmentation makes the model robust to variations
- The trained model is saved as `mask_detector.keras`

### 2. Real-Time Mask Detection

```bash
python detect_mask_video.py
```
A webcam window will open. Press `q` to exit.

- Loads the trained model and a pretrained Caffe-based face detector
- Captures video frames from your webcam
- Detects faces in each frame and classifies mask usage
- Displays results live with bounding boxes and confidence labels

---

## Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/mask_detector_week2.git
    cd mask_detector_week2
    ```

2. **Install dependencies:**
    ```bash
    pip install tensorflow keras opencv-python imutils scikit-learn numpy matplotlib
    ```

---

## Project Structure

```
mask_detector_week2/
│
├── train_mask_detector.py      # Training script for the mask detector model
├── detect_mask_video.py        # Real-time face mask detection using webcam
├── mask_detector.keras         # Trained mask classification model
├── plot.png                    # Training/validation loss & accuracy graph
├── face_detector/
│     ├── deploy.prototxt       # Caffe prototxt file for face detection
│     └── res10_300x300_ssd_iter_140000.caffemodel # Pretrained weights
├── README.md                   # Project overview and instructions
└── ...
```

---

## Results

- The system accurately classifies mask usage in live video from a webcam.
- Multiple faces can be detected and classified in each frame.
- Training and validation metrics (loss and accuracy) are visualized in `plot.png`.
- The trained model (`mask_detector.keras`) can be easily reused or deployed.

---

## Future Enhancements

- Expand the dataset with more diverse images (different lighting, mask types, backgrounds)
- Add support for detection from video files or images (not just webcam)
- Integrate with alarm or notification systems for compliance monitoring
- Optimize for edge devices (Raspberry Pi, mobile)
- Deploy as a web app or cloud service

---

## References

- [TensorFlow and Keras documentation](https://www.tensorflow.org/)
- [OpenCV documentation](https://docs.opencv.org/)
- [MobileNetV2 research paper](https://arxiv.org/abs/1801.04381)
- Mask detection datasets (publicly available)
- YouTube tutorials and guides on deep learning, transfer learning, and mask detection  
  (Including [CodeBasics](https://www.youtube.com/c/CodeBasics), [freeCodeCamp](https://www.youtube.com/c/Freecodecamp), [PyImageSearch](https://www.youtube.com/c/PyImageSearch), and more)

---

## License

This project is provided for educational purposes.  
Feel free to use, modify, and share with attribution.

---

## Author

**Hafsa Fatima**  

