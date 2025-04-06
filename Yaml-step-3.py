import os
import random
import shutil
from tkinter import Tk, filedialog

# Step 1: Use tkinter to let the user select directories
def select_directory(title):
    """Open a dialog to select a directory."""
    root = Tk()
    root.withdraw()  # Hide the main tkinter window
    
    # Ensure the dialog appears on top
    root.attributes('-topmost', True)
    root.lift()
    
    directory = filedialog.askdirectory(title=title)
    root.destroy()  # Explicitly destroy the root window after selection
    
    if not directory:
        raise ValueError("No directory selected. Exiting.")
    return directory

print("Please select the directory where you want to save the organized dataset.")
OUTPUT_DIR = select_directory("Select Output Directory")

# Constants
TRAIN_RATIO = 0.8  # 80% for training, 20% for validation

# Step 2: Collect class-specific image and label folders
CLASS_FOLDERS = []
CLASS_NAMES = []

print("\nYou will now be prompted to select folders for each class.")
while True:
    print("\nEnter details for a new class (or press 'Cancel' to stop):")
    class_name = input("Enter the name of the class (e.g., 'Babu', 'Gokul'): ")
    if not class_name.strip():
        if not CLASS_FOLDERS:
            raise ValueError("No classes were added. Exiting.")
        break
    
    print(f"Select the folder containing IMAGES for class '{class_name}':")
    images_dir = select_directory(f"Select Images Folder for Class '{class_name}'")
    
    print(f"Select the folder containing LABELS for class '{class_name}':")
    labels_dir = select_directory(f"Select Labels Folder for Class '{class_name}'")
    
    CLASS_FOLDERS.append((class_name, images_dir, labels_dir))
    CLASS_NAMES.append(class_name)
    print(f"Added class '{class_name}' with images from {images_dir} and labels from {labels_dir}")

# Step 3: Create output directory structure
try:
    os.makedirs(os.path.join(OUTPUT_DIR, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "images", "val"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "labels", "train"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "labels", "val"), exist_ok=True)
except OSError as e:
    print(f"Error creating directories: {e}")

# Step 4: Organize data into train and validation sets
all_images = []
all_labels = []

for class_id, (class_name, images_dir, labels_dir) in enumerate(CLASS_FOLDERS):
    if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
        raise ValueError(f"Missing 'images' or 'labels' directory for class '{class_name}'")
    
    # Collect images and assign class ID
    class_images = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    for img_file in class_images:
        label_file = os.path.splitext(img_file)[0] + ".txt"
        src_label_path = os.path.join(labels_dir, label_file)
        
        if not os.path.exists(src_label_path):
            print(f"Warning: Missing label file for image {img_file} in class '{class_name}'. Skipping...")
            continue
        
        # Update label file with the correct class ID
        with open(src_label_path, "r") as f:
            lines = f.readlines()
        
        updated_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                print(f"Warning: Invalid label format in file {src_label_path}. Skipping...")
                continue
            # Replace the class ID with the current class ID
            updated_line = f"{class_id} {' '.join(parts[1:])}\n"
            updated_lines.append(updated_line)
        
        # Write the updated label file to a temporary location
        temp_label_path = os.path.join(OUTPUT_DIR, "temp_labels", label_file)
        os.makedirs(os.path.dirname(temp_label_path), exist_ok=True)
        with open(temp_label_path, "w") as f:
            f.writelines(updated_lines)
        
        # Add to the list of files to process
        all_images.append((img_file, images_dir))
        all_labels.append((label_file, temp_label_path))

# Shuffle and split the dataset
random.shuffle(all_images)
split_index = int(len(all_images) * TRAIN_RATIO)
train_images = all_images[:split_index]
val_images = all_images[split_index:]

# Copy files to the output directory
def copy_files(image_files, split):
    for img_file, src_img_dir in image_files:
        try:
            # Copy image
            src_img_path = os.path.join(src_img_dir, img_file)
            dst_img_path = os.path.join(OUTPUT_DIR, "images", split, img_file)
            shutil.copy(src_img_path, dst_img_path)
            
            # Copy corresponding label
            label_file = os.path.splitext(img_file)[0] + ".txt"
            src_label_path = os.path.join(OUTPUT_DIR, "temp_labels", label_file)
            dst_label_path = os.path.join(OUTPUT_DIR, "labels", split, label_file)
            shutil.copy(src_label_path, dst_label_path)
        except Exception as e:
            print(f"Error processing file {img_file}: {e}")

copy_files(train_images, "train")
copy_files(val_images, "val")

# Clean up temporary label files
shutil.rmtree(os.path.join(OUTPUT_DIR, "temp_labels"), ignore_errors=True)

print("Dataset organization complete!")

# Step 5: Create dataset configuration file
dataset_config = f"""
train: {os.path.abspath(os.path.join(OUTPUT_DIR, 'images', 'train'))}
val: {os.path.abspath(os.path.join(OUTPUT_DIR, 'images', 'val'))}

nc: {len(CLASS_NAMES)}  # Number of classes
names: {CLASS_NAMES}    # Class names
"""

with open(os.path.join(OUTPUT_DIR, "dataset.yaml"), "w") as f:
    f.write(dataset_config)

print("Dataset configuration file created!")
