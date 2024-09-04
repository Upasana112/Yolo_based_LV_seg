import cv2
import pandas as pd
import os
from tqdm import tqdm

# Step 1: Define Directories and Read CSV
video_dir = 'videos'  # Directory containing the original videos
images_dir = 'data/images/train'  # Directory to save extracted frames
labels_dir = 'data/labels/train'  # Directory to save YOLO annotations
csv_path = 'annotations.csv'  # Path to CSV file

# Create directories 
os.makedirs(images_dir, exist_ok=True)
os.makedirs(labels_dir, exist_ok=True)

# Read the Annotations CSV File
annotations_df = pd.read_csv(csv_path)

#  Extract Frames and Generate Annotations
for idx, row in tqdm(annotations_df.iterrows(), total=len(annotations_df), desc="Processing Annotations"):
    # Extract information from CSV row
    filename = row['FileName']
    x1, y1, x2, y2 = row['X1'], row['Y1'], row['X2'], row['Y2']
    frame_number = int(row['Frame'])
    
    # Determine the video path
    video_path = os.path.join(video_dir, filename)
    
    # Ensure the video file exists
    if not os.path.exists(video_path):
        print(f"Warning: Video file {video_path} does not exist. Skipping...")
        continue
    
    # Capture the video and move to the specified frame
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    
    # Read the frame
    ret, frame = cap.read()
    if not ret:
        print(f"Warning: Could not read frame {frame_number} from {video_path}. Skipping...")
        cap.release()
        continue
    
    # Construct image and label filenames
    image_filename = f"{filename}_frame{frame_number}.jpg"
    label_filename = f"{filename}_frame{frame_number}.txt"
    
    # Save the frame as an image
    image_path = os.path.join(images_dir, image_filename)
    cv2.imwrite(image_path, frame)
    
    # Get image dimensions
    img_height, img_width = frame.shape[:2]
    
    # Convert (X1, Y1, X2, Y2) to YOLO format (x_center, y_center, width, height)
    x_center = ((x1 + x2) / 2) / img_width
    y_center = ((y1 + y2) / 2) / img_height
    bbox_width = (x2 - x1) / img_width
    bbox_height = (y2 - y1) / img_height
    
    # Class ID (0 for left ventricle)
    class_id = 0
    
    # YOLO annotation string
    yolo_annotation = f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}"
    
    # Save YOLO annotation to the corresponding label file
    label_path = os.path.join(labels_dir, label_filename)
    with open(label_path, 'w') as f:
        f.write(yolo_annotation)
    
    print(f"Annotation saved: {label_path}")
    
    # Release the video capture object
    cap.release()

print("Annotation generation complete!")