from ultralytics import YOLO

# Load the newly trained YOLOv8 weights
model = YOLO("yolov8n_br_flip_blur_results/weights/best.pt", task="detect")

# Export to ONNX
model.export(format="onnx", dynamic=False, simplify=True, opset=11, imgsz=640)