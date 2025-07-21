from ultralytics import YOLO
import matplotlib
import matplotlib.pyplot as plt

from yolo_foci.constants import model_path, model_name

matplotlib.use('Agg')

# Load the saved model
model = YOLO(model_path)

# Run inference on an image
inference_results = model("../../data/OCT5k/yolo/images/test/AMD Part1_AMD (3).E2E_2- 25- 2017 9- 13- 55 PM_Image 16.png")

from ultralytics import YOLO
import matplotlib
import matplotlib.pyplot as plt
import cv2
import numpy as np

from yolo_foci.constants import model_path, model_name

matplotlib.use('Agg')  # Use non-interactive backend if needed

# Load the saved model
model = YOLO(model_path)

# Run inference on an image
image_path = "C:/WORK/Dissertation/data/foci/images/val/0028.bmp"
inference_results = model(image_path)

# Extract prediction data
result = inference_results[0]
img = cv2.imread(image_path)
h, w = img.shape[:2]

# Convert to YOLO format and draw boxes
print("YOLO format predictions:")
for box in result.boxes:
    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # Bounding box corners
    cls_id = int(box.cls[0].item())
    conf = box.conf[0].item()

    # Convert to YOLO format
    x_center = ((x1 + x2) / 2) / w
    y_center = ((y1 + y2) / 2) / h
    width = (x2 - x1) / w
    height = (y2 - y1) / h

    # Print the annotation
    print(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    # Draw the box and label
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    label = f"Class {cls_id} ({conf:.2f})"
    cv2.putText(img, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

# Convert BGR (OpenCV) to RGB (matplotlib)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Display using matplotlib
plt.imshow(img_rgb)
plt.axis('off')
plt.title("Predictions with Class Labels")
plt.savefig("./test_example.png")
plt.close()