import tensorflow as tf
import cv2
import os
import numpy as np
from DeepLabV3APSS import DeepLabV3ASPP
from ReadDataSet import ReadDataset

label_color = {"head": [255, 0, 255], "shadow": [0, 255, 0]}
Predict_Read_Path = ".//Hellas_True//"
Predict_Save_Path = ".//PredictMask_Hellas//"
weights_path = ".//new_beight//DeepLabV3ASPP35000_96.472.ckpt"

model = DeepLabV3ASPP()
model.load_weights(weights_path)
for i, j, k in os.walk(Predict_Read_Path):  # 获取文件夹中的所有文件名称
    catalogList = k  # k是个列表，包含所有该目录下的文件名
catalogList = catalogList
i = 0
len_list = len(catalogList)
for file_name in catalogList:
    image = cv2.imread(Predict_Read_Path + file_name)
    org_h, org_w, org_c = image.shape
    image = cv2.resize(image, (256, 256))

    x = tf.convert_to_tensor([image], dtype=tf.float32) / 255
    out = model(x)

    out = tf.argmax(out, axis=-1).numpy()
    show_boart_background = np.zeros((out.shape[0], out.shape[1], out.shape[2], 3), dtype=np.uint8)
    for ind, type_object in enumerate(label_color.keys()):
        show_boart_background[out == ind + 1] = label_color[type_object]

    show_boart_background = np.squeeze(show_boart_background)
    show_boart_background = cv2.resize(show_boart_background, (org_w, org_h))
    cv2.imwrite(Predict_Save_Path + file_name, show_boart_background)

    print("\r", end="")
    print("Predict progress %d/%d : " % (i, len_list), "▮" * int((50 * i / len_list)),
          "▯" * int(50 - (50 * i / len_list)), "  %.2f %%" % (100 * i / len_list), end="")
    i += 1

