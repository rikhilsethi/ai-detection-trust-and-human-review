from ultralytics import YOLO
import json
import os
import numpy as np
from tqdm import tqdm

# Load model
model = YOLO('yolov8n.pt')

# Load COCO ground truth labels
print("Loading ground truth labels...")
with open('annotations/instances_val2017.json', 'r') as f:
    coco = json.load(f)

# Build a lookup: image_id -> list of ground truth boxes
gt_lookup = {}
for ann in coco['annotations']:
    img_id = ann['image_id']
    if img_id not in gt_lookup:
        gt_lookup[img_id] = []
    gt_lookup[img_id].append({
        'bbox': ann['bbox'],  # [x, y, width, height]
        'category_id': ann['category_id']
    })

# Build image id -> filename lookup
img_lookup = {img['id']: img['file_name'] for img in coco['images']}

# COCO to YOLO class mapping
coco_to_yolo = {
    1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 8:7, 9:8, 10:9,
    11:10, 13:11, 14:12, 15:13, 16:14, 17:15, 18:16, 19:17,
    20:18, 21:19, 22:20, 23:21, 24:22, 25:23, 27:24, 28:25,
    31:26, 32:27, 33:28, 34:29, 35:30, 36:31, 37:32, 38:33,
    39:34, 40:35, 41:36, 42:37, 43:38, 44:39, 46:40, 47:41,
    48:42, 49:43, 50:44, 51:45, 52:46, 53:47, 54:48, 55:49,
    56:50, 57:51, 58:52, 59:53, 60:54, 61:55, 62:56, 63:57,
    64:58, 65:59, 67:60, 70:61, 72:62, 73:63, 74:64, 75:65,
    76:66, 77:67, 78:68, 79:69, 80:70, 81:71, 82:72, 84:73,
    85:74, 86:75, 87:76, 88:77, 89:78, 90:79
}

def compute_iou(box1, box2):
    """Compute IoU between two boxes in [x,y,w,h] format"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Convert to [x1,y1,x2,y2]
    b1 = [x1, y1, x1+w1, y1+h1]
    b2 = [x2, y2, x2+w2, y2+h2]

    # Intersection
    ix1 = max(b1[0], b2[0])
    iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2])
    iy2 = min(b1[3], b2[3])

    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    intersection = (ix2-ix1) * (iy2-iy1)
    union = w1*h1 + w2*h2 - intersection
    return intersection / union

# Run detection on first 500 images for speed
results_data = []
image_ids = list(img_lookup.keys())[:500]

print(f"Running detection on {len(image_ids)} images...")

for img_id in tqdm(image_ids):
    img_file = os.path.join('val2017', img_lookup[img_id])
    if not os.path.exists(img_file):
        continue

    # Run YOLO
    results = model(img_file, verbose=False)
    gt_boxes = gt_lookup.get(img_id, [])

    for box in results[0].boxes:
        confidence = float(box.conf)
        class_id = int(box.cls)

        # Get predicted box in [x,y,w,h]
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        pred_box = [x1, y1, x2-x1, y2-y1]

        # Check if this matches any ground truth box
        correct = 0
        for gt in gt_boxes:
            gt_yolo_class = coco_to_yolo.get(gt['category_id'], -1)
            if gt_yolo_class == class_id:
                iou = compute_iou(pred_box, gt['bbox'])
                if iou >= 0.5:
                    correct = 1
                    break

        results_data.append({
            'confidence': confidence,
            'correct': correct,
            'class_id': class_id
        })

# Save results
import pandas as pd
df = pd.DataFrame(results_data)
df.to_csv('predictions.csv', index=False)
print(f"\nDone! Collected {len(df)} predictions.")
print(f"Correct: {df['correct'].sum()} | False Positives: {(df['correct']==0).sum()}")
print("Saved to predictions.csv")