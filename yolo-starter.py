import ultralytics
from ultralytics import YOLO

# Replace coco8.yaml with dataset path

# For training on GPU (or CPU if GPU unavailable):
# Load a model
model = YOLO("yolo11n.yaml")  # build a new model from YAML
model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)
model = YOLO("yolo11n.yaml").load("yolo11n.pt")  # build from YAML and transfer weights

# Train the model
results = model.train(data="Aerial Person Detection.v3-changed-labels.yolov11/data.yaml", epochs=50, imgsz=640)


# # For training on MPS:
# # Load a model
# model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)

# # Train the model with MPS
# results = model.train(data="Aerial Person Detection.v3-changed-labels.yolov11/data.yaml", epochs=100, imgsz=640, device="mps")

# # Run inference:
# results = model.predict(source="path/to/images_or_folder", imgsz=640, conf=0.25, save=True)