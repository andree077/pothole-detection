import cv2
import numpy as np
import tensorflow as tf

# Load the pre-trained object detection model

model = tf.saved_model.load("C:/Projects/CODEutsava/full_model.h5")

# Function to perform object detection on a frame
def detect_objects(frame):
    input_tensor = tf.convert_to_tensor(frame)
    detections = model(input_tensor)
    return detections

# Function to draw bounding boxes on detected objects
def draw_boxes(frame, detections, confidence_threshold=0.5):
    for detection in detections['detection_boxes'][0]:
        ymin, xmin, ymax, xmax = detection
        confidence = detections['detection_scores'][0][0]

        if confidence > confidence_threshold:
            h, w, _ = frame.shape
            x1, y1, x2, y2 = int(xmin * w), int(ymin * h), int(xmax * w), int(ymax * h)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Confidence: {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame

# Open the smartphone camera or webcam
cap = cv2.VideoCapture(0)  # Use 0 for the default camera

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection on the frame
    detections = detect_objects(frame)

    # Draw bounding boxes on detected objects
    frame_with_boxes = draw_boxes(frame, detections)

    cv2.imshow('Real-time Object Detection', frame_with_boxes)

    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()
