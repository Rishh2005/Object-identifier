import numpy as np
import time
import cv2
from utils import *

"""
Detects objects in web-cam video feed and shows bounding boxes with class labels.
"""

np.random.seed(42)

CONF = 0.5  # Confidence
THRESH = 0.3  # Threshold

# Loading class labels and YOLO model
labels, model = load_yolo("yolo-coco")
# Output layer names needed from YOLO
ln = model.getLayerNames()
ln = [ln[i[0] - 1] for i in model.getUnconnectedOutLayers()]
# Generating some colors for each class
colors = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")
# Accessing web-cam
cap = cv2.VideoCapture(0)

while True:
    # Reading in each frame as an image
    _, image = cap.read()
    # Image height (H) and width (W)
    (H, W) = image.shape[:2]

    # Generate blob from input image, do forward pass with YOLO detector, give bounding boxes and probabilities
    blob = cv2.dnn.blobFromImage(image, scalefactor=1 / 255, size=(416, 416), swapRB=True, crop=False)
    model.setInput(blob)
    start = time.time()
    layer_outputs = model.forward(ln)
    end = time.time()
    print(f"YOLO prediction took {np.round(end - start, 6)} seconds")

    boxes = []
    confidences = []
    class_ids = []
    # Iterate over each of layer_outputs and draw prediction and bbox on output image
    for i, output in enumerate(layer_outputs):
        # Iterate over each of the detections
        for detection in output:
            scores = detection[5:]  # Class probabilities
            classID = np.argmax(scores)  # Class IDs
            confidence = scores[classID]  # Confidence
            # Filtering out low confidence predictions
            if confidence > CONF:
                # Bounding box dimensions
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                # Inferring top right x, y from centered x, y (to draw box in cv2)
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                # Add to main lists
                boxes.append([x, y, int(width), int(height)])  # Bounding boxes
                confidences.append(float(confidence))  # Class confidence
                class_ids.append(classID)  # Class IDs

    # Applying non-max suppression to suppress weak overlapping bounding boxes
    ids = cv2.dnn.NMSBoxes(bboxes=boxes, scores=confidences, score_threshold=CONF, nms_threshold=THRESH)
    print(f"Detections kept after non-max suppression: {len(ids)}")

    # Ensuring at least one detection is present
    if len(ids) > 0:
        # Iterate over indexes
        for i in ids.flatten():
            # Bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            # Bounding box width, height
            (w, h) = (boxes[i][2], boxes[i][3])
            # Draw bounding box rectangle and label the image
            color = [int(c) for c in colors[class_ids[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = f"{labels[class_ids[i]]}: {np.round(confidences[i], 6)}"
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display output image
    cv2.imshow(f"Live object detection", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release web-cam & destroy windows
cap.release()
cv2.destroyAllWindows()
