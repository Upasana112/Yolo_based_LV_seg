import os
import shutil
from sklearn.model_selection import train_test_split

# Define directories
images_dir = 'data/images/train'  # Directory with all images
labels_dir = 'data/labels/train'  # Directory with all labels
train_images_dir = 'data/images/train_split'  # Directory to save training images after split
val_images_dir = 'data/images/val'  # Directory to save validation images after split
train_labels_dir = 'data/labels/train_split'  # Directory to save training labels after split
val_labels_dir = 'data/labels/val'  # Directory to save validation labels after split

# Create directories if they don't exist
os.makedirs(train_images_dir, exist_ok=True)
os.makedirs(val_images_dir, exist_ok=True)
os.makedirs(train_labels_dir, exist_ok=True)
os.makedirs(val_labels_dir, exist_ok=True)

# Get list of all images and corresponding label files
image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.jpg')])
label_files = sorted([f for f in os.listdir(labels_dir) if f.endswith('.txt')])

# Ensure matching image and label files
assert len(image_files) == len(label_files), "Mismatch between images and labels count."

# Split the data into training and validation sets
train_images, val_images, train_labels, val_labels = train_test_split(
    image_files, label_files, test_size=0.2, random_state=42
)

# Copy files to train and validation directories
for image_file, label_file in zip(train_images, train_labels):
    shutil.copy(os.path.join(images_dir, image_file), os.path.join(train_images_dir, image_file))
    shutil.copy(os.path.join(labels_dir, label_file), os.path.join(train_labels_dir, label_file))

for image_file, label_file in zip(val_images, val_labels):
    shutil.copy(os.path.join(images_dir, image_file), os.path.join(val_images_dir, image_file))
    shutil.copy(os.path.join(labels_dir, label_file), os.path.join(val_labels_dir, label_file))

print("Data has been split into train and validation sets.")
