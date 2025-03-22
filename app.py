import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO

# Step 1: Initialize variables
model = None
model_path = ""

# Step 2: Function to load the YOLOv8 model
def select_model():
    global model, model_path
    file_path = filedialog.askopenfilename(
        title="Select Model File",
        filetypes=[("Model Files", "*.pt"), ("All Files", "*.*")]
    )
    if not file_path:
        return  # User canceled the file dialog

    try:
        # Load the selected model
        model = YOLO(file_path)
        model_path = file_path
        messagebox.showinfo("Success", f"Model loaded successfully: {file_path}")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load model: {e}")

# Step 3: Function to perform object detection
def detect_objects(image_path):
    try:
        if model is None:
            raise ValueError("No model loaded. Please select a model first.")

        # Load the image using OpenCV
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Unable to load the image. Please check the file path.")

        # Perform inference
        results = model(image)

        # Parse results and draw bounding boxes
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()  # Bounding box coordinates
            scores = result.boxes.conf.cpu().numpy()  # Confidence scores
            class_ids = result.boxes.cls.cpu().numpy()  # Class IDs

            for box, score, class_id in zip(boxes, scores, class_ids):
                x1, y1, x2, y2 = map(int, box)
                label = f"{model.names[int(class_id)]} {score:.2f}"
                color = (0, 255, 0)  # Green bounding box

                # Draw bounding box and label
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Convert the image to RGB format for displaying in Tkinter
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image_rgb

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")
        return None

# Step 4: Function to handle image selection and display results
def open_image():
    if model is None:
        messagebox.showwarning("Warning", "Please select a model before opening an image.")
        return

    file_path = filedialog.askopenfilename(
        title="Select Image File",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp"), ("All Files", "*.*")]
    )
    if not file_path:
        return  # User canceled the file dialog

    # Perform object detection
    detected_image = detect_objects(file_path)
    if detected_image is None:
        return  # Error occurred during detection

    # Convert the image to a format compatible with Tkinter
    image_pil = Image.fromarray(detected_image)
    image_tk = ImageTk.PhotoImage(image_pil)

    # Update the GUI to display the image
    result_label.config(image=image_tk)
    result_label.image = image_tk  # Keep a reference to avoid garbage collection

# Step 5: Create the GUI
root = tk.Tk()
root.title("YOLOv8 Object Detection")

# Label to display instructions
instruction_label = tk.Label(root, text="1. Select a model file.\n2. Open an image to run object detection.")
instruction_label.pack(pady=10)

# Button to select the model
select_model_button = tk.Button(root, text="Select Model", command=select_model)
select_model_button.pack(pady=5)

# Button to open an image
open_button = tk.Button(root, text="Open Image", command=open_image)
open_button.pack(pady=5)

# Label to display the result image
result_label = tk.Label(root)
result_label.pack()

# Run the GUI main loop
root.mainloop()