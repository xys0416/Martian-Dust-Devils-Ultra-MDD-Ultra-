import tensorflow as tf
import cv2
import os
import numpy as np
from DeepLabV3APSS import DeepLabV3ASPP
from ReadDataSet import ReadDataset

# Color map for each segmentation class
label_color = {"head": [255, 0, 255], "shadow": [0, 255, 0]}

# Paths
Predict_Read_Path = ".//MDDD_Cascade_output_true//"   # Folder containing input images
Predict_Save_Path = ".//PredictMask//"                # Folder to save predicted masks
weights_path = ".//new_beight//DeepLabV3ASPP35000_96.472.ckpt"

# Build and load model
model = DeepLabV3ASPP()
model.load_weights(weights_path)

# Collect all file names in the read directory
catalogList = []
for dirpath, dirnames, filenames in os.walk(Predict_Read_Path):
    catalogList = filenames  # 'filenames' is a list of all files in the directory
    break  # Only need the top-level directory

total_files = len(catalogList)

# Process each image
for idx, file_name in enumerate(catalogList):
    # Read and resize image
    image = cv2.imread(os.path.join(Predict_Read_Path, file_name))
    if image is None:
        print(f"Warning: Unable to read image {file_name}, skipping.")
        continue
    org_h, org_w, _ = image.shape
    image = cv2.resize(image, (256, 256))

    # Prepare input tensor
    x = tf.convert_to_tensor([image], dtype=tf.float32) / 255.0

    # Predict segmentation mask
    out = model(x)
    out = tf.argmax(out, axis=-1).numpy()  # Shape: (1, 256, 256)

    # Create colored mask
    show_board_background = np.zeros((out.shape[1], out.shape[2], 3), dtype=np.uint8)
    for class_idx, class_name in enumerate(label_color.keys()):
        # class_idx+1 because 0 is background
        show_board_background[out[0] == (class_idx + 1)] = label_color[class_name]

    # Resize back to original size and save
    show_board_background = cv2.resize(show_board_background, (org_w, org_h))
    cv2.imwrite(os.path.join(Predict_Save_Path, file_name), show_board_background)

    # Progress bar
    progress = int(50 * idx / total_files)
    print("\rPredict progress %d/%d : " % (idx + 1, total_files),
          "▮" * progress, "▯" * (50 - progress),
          "  %.2f %%" % (100 * (idx + 1) / total_files), end="")

print("\nPrediction finished.")
