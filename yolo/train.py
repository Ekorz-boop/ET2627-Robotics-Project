import os
from ultralytics import YOLO

# TO ENABLE RANDOM BLUR INSTALL "albumentations"
# using pip install albumentations

def main():
    model = YOLO("yolov5n.pt", task="detect")
    model.train(
        data='yolo/data.yaml',
        device=0,
        workers=12,
        epochs=100,
        imgsz=640,
        batch=-1,
        optimizer='AdamW',
        patience=10,
        project=os.getcwd(), 
        name='yolov5n_br_flip_blur_results', # Change this to the desired output name
        exist_ok=False, # Overwrites/does not the folder if you run it multiple times

        # DATA AUGMENTATION PIPELINE
        hsv_v=0.25,
        fliplr=0.25
    )


if __name__ == "__main__":
    main()