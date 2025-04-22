import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as T
from PIL import Image
import cv2 # OpenCV for camera and drawing
import numpy as np
import time
import os

# --- Configuration ---
MODEL_PATH = 'retail_shelf_monitor_model_epochs5.pth'
# IMPORTANT: Define your class mapping (index to label name)
# Based on your previous output: {'__background__': 0, 'aqua': 1, 'chitato': 2, 'indomie': 3, 'pepsodent': 4, 'shampoo': 5, 'tissue': 6}
INT_TO_LABEL = {
    0: '__background__', 1: 'aqua', 2: 'chitato', 3: 'indomie',
    4: 'pepsodent', 5: 'shampoo', 6: 'tissue'
}
NUM_CLASSES = len(INT_TO_LABEL)
CONFIDENCE_THRESHOLD = 0.6 # Only show detections with confidence >= 60%
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Using device: {DEVICE}")

# --- Model Definition (Include the function from your training script) ---
def get_object_detection_model(num_classes):
    # Load a model pre-trained on COCO (using V2 weights if available and compatible)
    try:
        weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=weights)
    except AttributeError: # Fallback to V1 if V2 not available
        print("FasterRCNN V2 weights not found, falling back to V1.")
        weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# --- Load Model ---
print(f"Loading model from {MODEL_PATH}...")
if not os.path.exists(MODEL_PATH):
    print(f"Error: Model file not found at {MODEL_PATH}")
    exit()

try:
    model = get_object_detection_model(num_classes=NUM_CLASSES)
    # Load weights, ensuring compatibility between CPU/GPU saves
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval() # Set model to evaluation mode
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# --- Image Preprocessing Transform ---
transform = T.Compose([
    T.ToTensor() # Converts PIL Image (H, W, C) [0-255] to Tensor (C, H, W) [0.0-1.0]
])

# --- Initialize Webcam ---
# Use 0 for default camera. Change if you have multiple cameras.
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print("Starting real-time detection... Press 'q' to quit.")

# --- Real-time Detection Loop ---
while True:
    # 1. Capture Frame-by-Frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break

    # Keep a copy of the original frame for drawing later
    original_frame = frame.copy()

    # 2. Preprocess the Frame
    # Convert frame from OpenCV's BGR to PIL's RGB format
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)

    # Apply the transform
    input_tensor = transform(pil_image)
    # Add batch dimension (model expects list of tensors or batch)
    input_batch = [input_tensor.to(DEVICE)]

    # 3. Perform Inference
    with torch.no_grad(): # Turn off gradients for inference
        start_time = time.time()
        outputs = model(input_batch)
        end_time = time.time()

    # Process only the first output (since we sent a batch of 1)
    output = outputs[0]
    boxes = output['boxes'].cpu().numpy()
    labels = output['labels'].cpu().numpy()
    scores = output['scores'].cpu().numpy()

    # 4. Filter and Draw Detections
    detections_drawn = 0
    for box, label, score in zip(boxes, labels, scores):
        if score >= CONFIDENCE_THRESHOLD:
            detections_drawn += 1
            # Get class name
            class_name = INT_TO_LABEL.get(label, f"Unknown:{label}")

            # Convert box coordinates to integers
            xmin, ymin, xmax, ymax = map(int, box)

            # Draw bounding box
            cv2.rectangle(original_frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2) # Green box

            # Prepare label text
            label_text = f"{class_name}: {score:.2f}"

            # Put label text above the box
            (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(original_frame, (xmin, ymin - h - 5), (xmin + w, ymin - 5), (0, 255, 0), -1) # Green background
            cv2.putText(original_frame, label_text, (xmin, ymin - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA) # Black text

    # Display FPS (optional)
    fps = 1 / (end_time - start_time)
    cv2.putText(original_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # 5. Display the resulting frame
    cv2.imshow('Retail Shelf Detection (Press q to quit)', original_frame)

    # 6. Check for Quit Command
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Cleanup ---
print("Releasing resources...")
cap.release()
cv2.destroyAllWindows()
print("Done.")
