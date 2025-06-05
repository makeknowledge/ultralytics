from ultralytics import YOLO

# Load a model
model = YOLO("./yolo11n.pt")  # load a pretrained model (recommended for training)

# Train the model with 1 GPU
results = model.train(data="./coco8.yaml", epochs=100, imgsz=640)