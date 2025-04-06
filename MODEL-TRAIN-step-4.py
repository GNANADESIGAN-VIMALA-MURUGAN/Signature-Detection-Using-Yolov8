from ultralytics import YOLO
from tkinter import Tk, filedialog

# Step 1: Use tkinter to let the user select dataset.yaml files one by one
def select_file(title, filetypes):
    """Open a dialog to select a single file."""
    root = Tk()
    root.withdraw()  # Hide the main tkinter window
    root.lift()  # Bring the root window to the front
    root.attributes('-topmost', True)  # Ensure the window is on top
    file_path = filedialog.askopenfilename(title=title, filetypes=filetypes)
    if not file_path:
        print("No file selected. Skipping...")
    root.destroy()  # Destroy the hidden root window after selection
    return file_path

print("You will now be prompted to select dataset.yaml files one by one.")
print("Press 'Cancel' in the file dialog when you are done selecting files.")

DATASET_YAML_FILES = []
while True:
    print("\nPlease select a dataset.yaml file (or press 'Cancel' to stop):")
    yaml_file = select_file(
        title="Select Dataset YAML File",
        filetypes=[("YAML Files", "*.yaml *.yml"), ("All Files", "*.*")]
    )
    
    if not yaml_file:  # User pressed 'Cancel'
        break
    
    DATASET_YAML_FILES.append(yaml_file)
    print(f"Added: {yaml_file}")

if not DATASET_YAML_FILES:
    raise ValueError("No dataset files were selected. Exiting.")

# Step 2: Define training parameters
EPOCHS = 50  # Number of training epochs
BATCH_SIZE = 16  # Batch size for training
IMAGE_SIZE = 640  # Input image size
MODEL_NAME = "yolov8n.pt"  # Pre-trained YOLOv8 model (optional, comment out for training from scratch)
OUTPUT_DIR = "runs/train"  # Directory to save training results

# Step 3: Load the pre-trained YOLOv8 model (or initialize a new one)
try:
    print("Loading pre-trained model...")
    model = YOLO(MODEL_NAME)  # Use this line if you have a pre-trained model
except FileNotFoundError:
    print("No pre-trained model found. Initializing a new model...")
    model = YOLO("yolov8n.yaml")  # Use this line for training from scratch

# Step 4: Check for GPU availability
import torch
if torch.cuda.is_available():
    print(f"GPU detected: {torch.cuda.get_device_name(0)}")
    DEVICE = "0"  # Use the first GPU
else:
    print("No GPU detected. Falling back to CPU.")
    DEVICE = "cpu"

# Step 5: Train the model for each dataset
for i, yaml_file in enumerate(DATASET_YAML_FILES):
    print(f"\nTraining on dataset {i + 1}/{len(DATASET_YAML_FILES)}: {yaml_file}")
    
    # Define a unique experiment name for each dataset
    experiment_name = f"exp_{i + 1}"
    
    print("Starting training...")
    results = model.train(
        data=yaml_file,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMAGE_SIZE,
        device=DEVICE,  # Use GPU if available, otherwise fall back to CPU
        project="yolov8_training2",
        name=experiment_name,
        exist_ok=True  # Overwrite existing experiment directory if it exists
    )

    print(f"Training complete for dataset {i + 1}: {yaml_file}")

print("\nAll training processes are complete!")
