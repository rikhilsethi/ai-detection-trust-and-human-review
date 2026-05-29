from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Load the model
model = YOLO('yolov8n.pt')

# Run detection on a sample image
results = model(r"C:\Users\rikhi\Desktop\Virtual Studio Projects\uncertainty-cv-project\Baddies.jpeg")

# Save the result image with bounding boxes drawn
results[0].save(filename='result.jpg')

# Print what was detected
print("\n--- DETECTIONS ---")
for box in results[0].boxes:
    class_id = int(box.cls)
    class_name = results[0].names[class_id]
    confidence = float(box.conf)
    print(f"Object: {class_name}, Confidence: {confidence:.2f}")

# Show the result image
img = mpimg.imread('result.jpg')
plt.figure(figsize=(12, 8))
plt.imshow(img)
plt.axis('off')
plt.title('YOLOv8 Detection Results')
plt.show()