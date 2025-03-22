import os
import cv2
import numpy as np
import random
import zipfile
from tkinter import filedialog, Tk

# Function to open file dialog for selecting multiple images
def select_images():
    root = Tk()
    root.withdraw()  # Hide the root window
    image_paths = filedialog.askopenfilenames(
        title="Select Images",
        filetypes=[("Image files", "*.jpeg;*.jpg;*.png")]
    )
    root.destroy()  # Destroy the root window
    return image_paths

# Function to open file dialog for selecting output directory
def select_output_directory():
    root = Tk()
    root.withdraw()  # Hide the root window
    output_dir = filedialog.askdirectory(title="Select Output Directory")
    root.destroy()  # Destroy the root window
    return output_dir

# Function to apply color filters (RGB)
def apply_color_filters(image):
    B, G, R = cv2.split(image)
    red_image = cv2.merge([np.zeros_like(B), np.zeros_like(G), R])
    green_image = cv2.merge([np.zeros_like(B), G, np.zeros_like(R)])
    blue_image = cv2.merge([B, np.zeros_like(G), np.zeros_like(R)])
    return red_image, green_image, blue_image

# Function to apply CMY filters
def apply_cmy_filters(image):
    img_float = image.astype(float) / 255.0
    K = 1 - np.max(img_float, axis=2)
    C = (1 - img_float[..., 2] - K) / (1 - K + 1e-5)
    M = (1 - img_float[..., 1] - K) / (1 - K + 1e-5)
    Y = (1 - img_float[..., 0] - K) / (1 - K + 1e-5)
    
    cyan_image = np.uint8(cv2.merge([255 * (1 - C), 255 * (1 - K), 255 * (1 - K)]))
    magenta_image = np.uint8(cv2.merge([255 * (1 - K), 255 * (1 - M), 255 * (1 - K)]))
    yellow_image = np.uint8(cv2.merge([255 * (1 - K), 255 * (1 - K), 255 * (1 - Y)]))
    
    return cyan_image, magenta_image, yellow_image

# Function to apply HSV filter
def apply_hsv_filter(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Function to apply grayscale conversion
def apply_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Function to apply black-and-white (thresholding) conversion
def apply_black_and_white(image):
    gray = apply_grayscale(image)
    _, bw_image = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    return bw_image

# Function to apply Gaussian blur
def apply_gaussian_blur(image):
    return cv2.GaussianBlur(image, (7, 7), 0)

# Function to apply channel blur (blurring specific channels)
def apply_channel_blur(image):
    B, G, R = cv2.split(image)
    B = cv2.GaussianBlur(B, (15, 15), 0)
    G = cv2.GaussianBlur(G, (15, 15), 0)
    R = cv2.GaussianBlur(R, (15, 15), 0)
    return cv2.merge([B, G, R])

# Function to apply zoom blur
def apply_zoom_blur(image):
    height, width = image.shape[:2]
    zoom_factor = 1.2
    zoomed = cv2.resize(image, None, fx=zoom_factor, fy=zoom_factor)
    crop_x = int((zoomed.shape[1] - width) / 2)
    crop_y = int((zoomed.shape[0] - height) / 2)
    zoomed = zoomed[crop_y:crop_y + height, crop_x:crop_x + width]
    return cv2.addWeighted(image, 0.6, zoomed, 0.4, 0)

# Function to apply directional blur (motion blur effect)
def apply_directional_blur(image):
    kernel_size = 15
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
    kernel /= kernel_size
    return cv2.filter2D(image, -1, kernel)

# Function to apply defocus blur
def apply_defocus_blur(image):
    kernel_size = 15
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

# Function to save augmented images in a zip file
def save_augmented_images_to_zip(image, base_filename, output_folder, ratio):
    zip_path = os.path.join(output_folder, f"{base_filename}.zip")
    
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        # Apply all filters and save augmented images
        red_image, green_image, blue_image = apply_color_filters(image)
        cyan_image, magenta_image, yellow_image = apply_cmy_filters(image)
        hsv_image = apply_hsv_filter(image)
        grayscale_image = apply_grayscale(image)
        bw_image = apply_black_and_white(image)
        blurred_image = apply_gaussian_blur(image)
        channel_blur_image = apply_channel_blur(image)
        zoom_blur_image = apply_zoom_blur(image)
        directional_blur_image = apply_directional_blur(image)
        defocus_blur_image = apply_defocus_blur(image)

        augmented_images = [red_image, green_image, blue_image, cyan_image, magenta_image, yellow_image,
                            hsv_image, grayscale_image, bw_image, blurred_image, channel_blur_image,
                            zoom_blur_image, directional_blur_image, defocus_blur_image]
        suffixes = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'hsv', 'grayscale', 'bw',
                    'blur', 'channel_blur', 'zoom_blur', 'directional_blur', 'defocus_blur']
        
        count = 0
        for i in range(ratio):
            random_suffixes = random.sample(suffixes, len(suffixes))[:len(suffixes)]
            for suffix, img in zip(random_suffixes, augmented_images):
                temp_filename = f"{base_filename}_{suffix}_{i+1}.jpeg"
                temp_image_path = os.path.join(output_folder, temp_filename)
                cv2.imwrite(temp_image_path, img)
                zipf.write(temp_image_path, temp_filename)
                os.remove(temp_image_path)  # Remove the temporary file after adding to zip
                count += 1  # Increment the count for each augmented image
    return count

# Main function to handle user interaction
def main():
    image_paths = select_images()
    output_folder = select_output_directory()  # Get output directory from user
    
    if not output_folder:  # Check if user selected a directory
        print("No output directory selected. Exiting...")
        return
    
    ratio = int(input("Enter the augmentation ratio (e.g., 2 for 2x): "))
    
    total_images_generated = 0
    for img_path in image_paths:
        image = cv2.imread(img_path)
        if image is None:
            continue
        base_filename = os.path.splitext(os.path.basename(img_path))[0]
        count = save_augmented_images_to_zip(image, base_filename, output_folder, ratio)
        total_images_generated += count

    print(f"Augmented images have been saved in individual zip files.")
    print(f"Total number of augmented images generated: {total_images_generated}")

if __name__ == "__main__":
    main()
