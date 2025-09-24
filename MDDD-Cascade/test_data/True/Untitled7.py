#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import random
from PIL import Image

# 设置参数
input_dir = 'C:/Users/2230021572/Desktop/predictTrue'  # 输入输出文件夹
a = 130  # 需要旋转的图像数量
b = 100   # 需要拉伸/缩小的图像数量
min_stretch_ratio = 0.8
max_stretch_ratio = 2.0

# 获取输入文件夹中的所有 jpg 图像
image_files = [f for f in os.listdir(input_dir) if f.endswith('.jpg')]

# 随机选择 a 张图像进行旋转
rotated_images = random.sample(image_files, a)

# 对选择的图像进行旋转并保存
for i, image_file in enumerate(rotated_images):
    image_path = os.path.join(input_dir, image_file)
    image = Image.open(image_path)
    for angle in [90, 180, 270]:
        rotated_image = image.rotate(angle)
        save_name = f"{os.path.splitext(image_file)[0]}_rotate{angle}_{i+1}.jpg"
        rotated_image.save(os.path.join(input_dir, save_name))

# 从原始和旋转后的图像中随机选择 b 张进行拉伸/缩小
stretched_images = random.sample(image_files + rotated_images, b)

# 对选择的图像进行拉伸/缩小并保存
for i, image_file in enumerate(stretched_images):
    image_path = os.path.join(input_dir, image_file)
    image = Image.open(image_path)
    for direction in ['horizontal', 'vertical']:
        stretch_ratio = random.uniform(min_stretch_ratio, max_stretch_ratio)
        if direction == 'horizontal':
            stretched_image = image.resize((int(image.width * stretch_ratio), image.height), resample=Image.BILINEAR)
        else:
            stretched_image = image.resize((image.width, int(image.height * stretch_ratio)), resample=Image.BILINEAR)
        save_name = f"{os.path.splitext(image_file)[0]}_{direction}_{stretch_ratio:.2f}_{i+1}.jpg"
        stretched_image.save(os.path.join(input_dir, save_name))

print("图像处理完成!")


# In[ ]:




