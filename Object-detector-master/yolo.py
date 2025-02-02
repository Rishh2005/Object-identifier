import numpy as np
import time
import cv2
import os
from utils import *

np.random.seed(42)

CONF = 0.5  # Confidence
THRESH = 0.3  # Threshold
MODEL_DIR = "yolo-coco"
IMG_DIR = "images"

# Loading class labels and YOLO model
labels, model = load_yolo(MODEL_DIR)
# All the layer names of the YOLO model
ln = model.getLayerNames()
# Only the output layer name(s)
out_layers = [ln[i[0] - 1] for i in model.getUnconnectedOutLayers()]
# Generating some random colors for each class
colors = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")

MODE = input("""Select object detection mode:
- images
- webcam
>>> """).lower()

if MODE in ["images", "image"]:
    file_names = os.listdir(IMG_DIR)
    file_paths = [os.path.join(IMG_DIR, file) for file in os.listdir(IMG_DIR)]
    # Iterate over all files in image directory
    for image_path in file_paths:
        print("-" * 40)
        print(f"Processing: '{image_path}'")
        print("-" * 40)
        # Reading in image as numpy array
        image = cv2.imread(image_path)
        # Image height and width
        (height, width) = image.shape[:2]

        # # # OBJECT DETECTION # # #
        layer_outputs = detect_objects(image, model, out_layers)
        boxes, confidences, class_ids = predict_bboxes(layer_outputs, width, height)
        # Applying non-max suppression to suppress weak overlapping bounding boxes
        ids = cv2.dnn.NMSBoxes(bboxes=boxes, scores=confidences, score_threshold=CONF, nms_threshold=THRESH)
        print(f"Detections kept after non-max suppression: {len(ids)}")
        # Drawing bounding boxes
        image = draw_boxes(image, ids, boxes, confidences, colors, class_ids, labels)

        # Display output image
        cv2.imshow(f"Image: {os.path.split(image_path)[1]}", image)
        cv2.waitKey(0)
elif MODE in ["webcam", "web-cam"]:
    WRITE = True if str(input("Write detection video to file? (y/n)\n>>> ")).lower() == "y" else False
    FPS = int(input("Enter FPS (frames per second) for recording: "))
    if WRITE:
        print("[INFO] DETECTIONS WILL BE SAVED TO FILE!")
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out = cv2.VideoWriter("detections.avi", fourcc, FPS, (640, 480))
    # Capture web-cam feed
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        f_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        f_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print("\nVideo playback FPS: \t", fps)
        print(f"Frame shape: {f_w}x{f_h}")
        print("-"*35)
        # Reading in each frame as an image
        _, image = cap.read()
        # Image height and width
        (height, width) = image.shape[:2]

        # # # OBJECT DETECTION # # #
        layer_outputs = detect_objects(image, model, out_layers)
        boxes, confidences, class_ids = predict_bboxes(layer_outputs, width, height)
        # Applying non-max suppression to suppress weak overlapping bounding boxes
        ids = cv2.dnn.NMSBoxes(bboxes=boxes, scores=confidences, score_threshold=CONF, nms_threshold=THRESH)
        print(f"Detections kept after non-max suppression: {len(ids)}")
        # Drawing bounding boxes
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

        # If we are supposed to write detections to file
        if WRITE:
            out.write(image)

        # Press Q to quit program
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()   # Release video capture
else:
    print("INVALID INPUT!")

cv2.destroyAllWindows()
