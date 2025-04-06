import os
import cv2
from tkinter import Tk
from tkinter.filedialog import askdirectory

drawing = False
ix, iy = -1, -1
boxes = []
class_id = None


def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, boxes

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            temp_image = param["display_image"].copy()
            cv2.rectangle(temp_image, (ix, iy), (x, y), (0, 0, 255), 2)
            cv2.imshow("Draw Bounding Boxes", temp_image)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        # Scale the bounding box back to the original image size
        scale_x = param["original_width"] / param["display_width"]
        scale_y = param["original_height"] / param["display_height"]
        original_box = (
            int(ix * scale_x),
            int(iy * scale_y),
            int(x * scale_x),
            int(y * scale_y),
        )
        boxes.append(original_box)
        cv2.rectangle(param["display_image"], (ix, iy), (x, y), (0, 0, 255), 2)
        cv2.imshow("Draw Bounding Boxes", param["display_image"])


def save_labels_to_txt(image_name, boxes, output_folder, image_width, image_height, class_id):
    output_file = os.path.join(output_folder, f"{os.path.splitext(image_name)[0]}.txt")
    with open(output_file, 'w') as f:
        for box in boxes:
            x1, y1, x2, y2 = box
            x_center = ((x1 + x2) / 2) / image_width
            y_center = ((y1 + y2) / 2) / image_height
            width = (x2 - x1) / image_width
            height = (y2 - y1) / image_height
            f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
    print(f"Labels saved to: {output_file}")


def upload_folder():
    Tk().withdraw()
    folder_path = askdirectory(title="Select Folder Containing Images")
    return folder_path


def main():
    global class_id
    folder_path = upload_folder()
    if not folder_path:
        print("No folder selected. Exiting.")
        return

    try:
        class_id = int(input("Enter the class ID for this dataset: "))
    except ValueError:
        print("Invalid class ID. It must be an integer. Exiting.")
        return

    print(f"Annotating images in folder: {folder_path}")
    print(f"Class ID: {class_id}")

    labels_folder = os.path.join(folder_path, "labels")
    os.makedirs(labels_folder, exist_ok=True)

    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    if not image_files:
        print("No valid images found in the folder. Exiting.")
        return

    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        original_image = cv2.imread(image_path)
        if original_image is None:
            print(f"Error: Could not load {image_file}. Skipping.")
            continue

        # Resize the image for display
        max_display_width = 1280
        max_display_height = 720
        original_height, original_width = original_image.shape[:2]
        scale_x = original_width / max_display_width
        scale_y = original_height / max_display_height
        scale = max(scale_x, scale_y)
        if scale > 1:
            display_width = int(original_width / scale)
            display_height = int(original_height / scale)
        else:
            display_width, display_height = original_width, original_height

        display_image = cv2.resize(original_image, (display_width, display_height))

        global boxes
        boxes = []
        cv2.namedWindow("Draw Bounding Boxes")
        cv2.setMouseCallback(
            "Draw Bounding Boxes",
            draw_rectangle,
            param={
                "display_image": display_image,
                "original_width": original_width,
                "original_height": original_height,
                "display_width": display_width,
                "display_height": display_height,
            },
        )

        print(f"Annotating: {image_file}. Draw rectangles using your mouse. Press 's' to save or 'q' to skip.")
        while True:
            cv2.imshow("Draw Bounding Boxes", display_image)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('s'):
                save_labels_to_txt(image_file, boxes, labels_folder, original_width, original_height, class_id)
                break

            elif key == ord('q'):
                print(f"Skipped: {image_file}")
                break

        cv2.destroyAllWindows()

    print("Annotation completed for all images in the folder.")


if __name__ == "__main__":
    main()
