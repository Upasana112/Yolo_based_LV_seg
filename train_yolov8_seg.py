from ultralytics import YOLO

model = YOLO('yolov8n.pt')  

results = model.train(
    data='yolo_config.yaml',  # Path to the data configuration file
    epochs=50,  # Number of epochs to train for
    imgsz=128,  # Image size for training
    batch=16,  # Batch size
    workers=4,  # Number of workers for data loading
    name='yolov8_seg_left_ventricle',  # Name of the training run
    project='yolov8_seg_training',  # Folder to save training results
    device='cpu'  
)

#  Evaluate the model on the validation dataset after training
metrics = model.val()

# Export the trained model to different formats (e.g., ONNX, CoreML, etc.)
model.export(format='onnx')  # Export the model to ONNX format
