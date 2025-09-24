import numpy as np
import cv2
import os

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


model = ResNet([16, 16, 16], 2)
model.load_weights("/home/xu/xys/DustStormClassificationDataset/new_beight/Resnet_Epo_5000_128_126.ckpt")

path = "/home/xu/xys/MDDD/Hellas_predict/"                # 预测文件夹
predict_True_path = "/home/xu/xys/DustStormClassificationDataset/predict_True/"      # 正确文件夹
predict_False_path = "/home/xu/xys/DustStormClassificationDataset/predict_False/"    # 错误文件夹

for i, j, k in os.walk(path):
    catalogList = k
fList = catalogList
fList = [path + fList[ind] for ind in range(len(fList))]

i = 0
len_list = len(fList)

for path_singele in fList:

    print("\r", end="")
    print("Predict progress %d/%d : "%(i, len_list), "▮" * int((50*i/len_list)), "▯"*int(50-(50*i/len_list)),  "  %.2f %%" %(100*i/len_list) , end="")
    i += 1

    image = cv2.imread(path_singele)
    if image is None:
        continue

    image = image_resize(image)

    image = tf.convert_to_tensor(image)
    image = tf.cast(image, tf.float32) / 255
    image = tf.expand_dims(image, axis=0)
    out = model(image)
    out = tf.argmax(out, axis=-1)
    image_name = path_singele.split("/")[-1]

    if out.numpy()[0] == 1:
        if os.path.join(predict_True_path, image_name) != path_singele:
            os.system('cp %s %s' % (path_singele, os.path.join(predict_True_path, image_name)))
    else:
        if os.path.join(predict_False_path, image_name) != path_singele:
            os.system('cp %s %s' % (path_singele, os.path.join(predict_False_path, image_name)))