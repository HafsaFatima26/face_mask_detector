from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os

def mask_prediction_on_frame(frame, face_net, mask_net):
    frame_height, frame_width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()
    detected_faces = []
    coordinates = []
    mask_outputs = []
    for i in range(0, detections.shape[2]):
        conf = detections[0, 0, i, 2]
        if conf > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([frame_width, frame_height, frame_width, frame_height])
            (x1, y1, x2, y2) = box.astype("int")
            (x1, y1) = (max(0, x1), max(0, y1))
            (x2, y2) = (min(frame_width - 1, x2), min(frame_height - 1, y2))
            face_region = frame[y1:y2, x1:x2]
            face_region = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
            face_region = cv2.resize(face_region, (224, 224))
            face_region = img_to_array(face_region)
            face_region = preprocess_input(face_region)
            detected_faces.append(face_region)
            coordinates.append((x1, y1, x2, y2))
    if len(detected_faces) > 0:
        detected_faces = np.array(detected_faces, dtype="float32")
        mask_outputs = mask_net.predict(detected_faces, batch_size=32)
    return (coordinates, mask_outputs)

proto_file = r"face_detector\deploy.prototxt"
weights_file = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
face_model = cv2.dnn.readNet(proto_file, weights_file)
mask_model = load_model("mask_detector.keras")
print("[INFO] Video stream initiated...")
video_capture = VideoStream(src=0).start()

while True:
    frame = video_capture.read()
    frame = imutils.resize(frame, width=400)
    (boxes, results) = mask_prediction_on_frame(frame, face_model, mask_model)
    for (rect, pred) in zip(boxes, results):
        (x1, y1, x2, y2) = rect
        (mask_score, no_mask_score) = pred
        label = "Mask" if mask_score > no_mask_score else "No Mask"
        color_draw = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        label = "{}: {:.2f}%".format(label, max(mask_score, no_mask_score) * 100)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color_draw, 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color_draw, 2)
    cv2.imshow("Live Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
video_capture.stop()
