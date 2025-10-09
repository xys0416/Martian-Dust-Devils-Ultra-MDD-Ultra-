import numpy as np
import cv2
import os
import time  
import shutil 
import tensorflow as tf
from Resnet import ResNet

def image_resize(img):
    ih, iw = img.shape[:2]
    w, h = (256, 256)

    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)
    dst = cv2.resize(img, (nw, nh))
    board = np.zeros((h, w, 3), dtype=np.uint8)

    w_start = (w - nw) // 2
    h_start = (h - nh) // 2
    board[h_start:h_start + nh, w_start:w_start + nw] = dst
    return board

# Load model
model = ResNet([16, 16, 16], 2)
model.load_weights("/home/xu/xys/DustStormClassificationDataset/new_beight/Resnet_Epo_5000_128_126.ckpt")

# Fix path format - use forward slashes and ensure path is correct
path = "./MDDD_output/"                # Folder to predict
predict_True_path = "./MDDD_Cascade_output_true/"      # Correct predictions folder
predict_False_path = "./MDDD_Cascade_output_false/"    # Incorrect predictions folder

# Ensure target directories exist
os.makedirs(predict_True_path, exist_ok=True)
os.makedirs(predict_False_path, exist_ok=True)

# Get all image file paths - directly read files in the specified directory
fList = []
valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')

# Directly read files in the path directory, no sub-directory traversal
if os.path.exists(path):
    for file in os.listdir(path):
        if file.lower().endswith(valid_extensions):
            full_path = os.path.join(path, file)
            # Check if file exists and is readable
            if os.path.isfile(full_path) and os.access(full_path, os.R_OK):
                fList.append(full_path)
            else:
                print(f"File is unreadable or does not exist: {full_path}")

print(f"Found {len(fList)} valid image files")

if len(fList) == 0:
    print("No processable image files found, please check path and file permissions")
    exit(1)

total_start_time = time.time()
frame_times = []
processed_count = 0

print("Starting image processing...")

for i, path_single in enumerate(fList):
    frame_start_time = time.time()

    # Update progress bar
    progress = int(50 * i / len(fList))
    print(f"\rPredict progress {i}/{len(fList)} : {'▮' * progress}{'▯' * (50 - progress)}  {100 * i / len(fList):.2f}%", end="")

    # Read image
    try:
        image = cv2.imread(path_single)
        if image is None:
            print(f"\nWarning: Unable to read image {path_single}, skipping")
            continue
            
        # Check if image is valid
        if image.size == 0:
            print(f"\nWarning: Image {path_single} is empty, skipping")
            continue

        # Preprocessing
        image_resized = image_resize(image)
        image_tensor = tf.convert_to_tensor(image_resized)
        image_tensor = tf.cast(image_tensor, tf.float32) / 255
        image_tensor = tf.expand_dims(image_tensor, axis=0)
        
        # Predict
        out = model(image_tensor)
        out = tf.argmax(out, axis=-1)
        
        # Get filename (keep original name)
        image_name = os.path.basename(path_single)
        
        # Copy to corresponding directory based on prediction result
        if out.numpy()[0] == 1:
            dest_path = os.path.join(predict_True_path, image_name)
        else:
            dest_path = os.path.join(predict_False_path, image_name)
        
        # Copy file (keep original filename)
        shutil.copy2(path_single, dest_path)
        processed_count += 1
        
    except Exception as e:
        print(f"\nError processing image {path_single}: {str(e)}")
        continue
    
    frame_time = time.time() - frame_start_time
    frame_times.append(frame_time)

# Performance statistics
total_time = time.time() - total_start_time
avg_frame_time = sum(frame_times) / len(frame_times) if frame_times else 0
fps = len(frame_times) / total_time if total_time > 0 else 0

print("\n\n" + "="*50)
print("Performance Statistics:")
print(f"Total images: {len(fList)}")
print(f"Successfully processed: {processed_count}")
print(f"Total time: {total_time:.2f} seconds")
print(f"Average frame processing time: {avg_frame_time:.4f} seconds")
print(f"Processing speed: {fps:.2f} frames/second (FPS)")
if frame_times:
    print(f"Fastest frame: {min(frame_times):.4f} seconds")
    print(f"Slowest frame: {max(frame_times):.4f} seconds")
else:
    print("No valid processing data")
print("="*50)
