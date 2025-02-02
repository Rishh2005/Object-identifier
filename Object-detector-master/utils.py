def load_yolo(directory):
    """
    Takes an input string specifying the directory of the YOLO weights, config and class
    label files and returns a tuple containing a list of class classes and the model.
    """
    # Class classes
    class_path = f"{directory}/coco.names"
    print("="*30)
    print("Running object detector")
    print("="*30)
    print(f"Loading class labels from '{class_path}'...")
    # Loading class labels
    classes = open(class_path).read().strip().split("\n")
    print(f"Contains {len(classes)} different classes.")

    from cv2.dnn import readNetFromDarknet
    # Path to model configuration and weights
    config_path = f"{directory}/yolov2.cfg"
    weight_path = f"{directory}/yolov2.weights"

    print(f"Loading YOLO model (config: '{config_path}', weights: '{weight_path}')...")
    # Loading model
    net = readNetFromDarknet(cfgFile=config_path, darknetModel=weight_path)

    return classes, net


def detect_objects(image, model, out_layers):
    from cv2.dnn import blobFromImage
    import time
    # Generate blob from input image, do forward pass with YOLO detector, give bounding boxes and probabilities
    blob = blobFromImage(image, scalefactor=1 / 255, size=(416, 416), swapRB=True, crop=False)
    model.setInput(blob)
    start = time.time()
    layer_outputs = model.forward(out_layers)
    end = time.time()
    print(f"YOLO prediction took {round(end - start, 6)} seconds")
    return layer_outputs


def predict_bboxes(layer_outputs, img_width, img_height, conf=0.5):
    from numpy import argmax, array
    boxes = []
    confidences = []
    class_ids = []
    # Iterate over each of layer_outputs and draw prediction and bbox on output image
    for i, output in enumerate(layer_outputs):
        print(f"Number of detections in output layer {i}: {len(output)}")
        # Iterate over each of the detections
        for detection in output:
            scores = detection[5:]  # Class probabilities
            classID = argmax(scores)  # Class IDs
            confidence = scores[classID]  # Confidence
            # Filtering out low confidence predictions
            if confidence > conf:
                # Bounding box dimensions
                box = detection[0:4] * array([img_width, img_height, img_width, img_height])
                (centerX, centerY, width, height) = box.astype("int")
                # Inferring top right x, y from centered x, y (to draw box in cv2)
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                # Add to main lists
                boxes.append([x, y, int(width), int(height)])  # Bounding boxes
                confidences.append(float(confidence))  # Class confidence
                class_ids.append(classID)  # Class IDs

    return boxes, confidences, class_ids


def draw_boxes(image, ids, boxes, confidences, colors, class_ids, labels):
    from cv2 import rectangle, putText, FONT_HERSHEY_SIMPLEX
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
            rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = f"{labels[class_ids[i]]}: {round(confidences[i], 6)}"
            putText(image, text, (x, y - 5), FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return image
