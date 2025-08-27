from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.pt")

# Train model with bakery dataset
results = model.train(data='Bakery-Detection/data.yaml', epochs=100, imgsz=640, batch=128)
