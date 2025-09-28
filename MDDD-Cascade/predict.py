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

# 加载模型
model = ResNet([16, 16, 16], 2)
model.load_weights("/home/xu/xys/DustStormClassificationDataset/new_beight/Resnet_Epo_5000_128_126.ckpt")

# 修复路径格式 - 使用正斜杠并确保路径正确
path = "./hirise_output/"                # 预测文件夹
predict_True_path = "./hirise_output_true/"      # 正确文件夹
predict_False_path = "./hirise_output_false/"    # 错误文件夹

# 确保目标目录存在
os.makedirs(predict_True_path, exist_ok=True)
os.makedirs(predict_False_path, exist_ok=True)

# 获取所有图片文件路径 - 直接读取指定目录下的文件
fList = []
valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')

# 直接读取path目录下的文件，不遍历子目录
if os.path.exists(path):
    for file in os.listdir(path):
        if file.lower().endswith(valid_extensions):
            full_path = os.path.join(path, file)
            # 检查文件是否确实存在且可读
            if os.path.isfile(full_path) and os.access(full_path, os.R_OK):
                fList.append(full_path)
            else:
                print(f"文件不可读或不存在: {full_path}")

print(f"找到 {len(fList)} 个有效的图片文件")

if len(fList) == 0:
    print("没有找到可处理的图片文件，请检查路径和文件权限")
    exit(1)

total_start_time = time.time()
frame_times = []
processed_count = 0

print("开始处理图片...")

for i, path_single in enumerate(fList):
    frame_start_time = time.time()

    # 更新进度条
    progress = int(50 * i / len(fList))
    print(f"\rPredict progress {i}/{len(fList)} : {'▮' * progress}{'▯' * (50 - progress)}  {100 * i / len(fList):.2f}%", end="")

    # 读取图片
    try:
        image = cv2.imread(path_single)
        if image is None:
            print(f"\n警告: 无法读取图片 {path_single}，跳过")
            continue
            
        # 检查图片是否有效
        if image.size == 0:
            print(f"\n警告: 图片 {path_single} 为空，跳过")
            continue

        # 预处理
        image_resized = image_resize(image)
        image_tensor = tf.convert_to_tensor(image_resized)
        image_tensor = tf.cast(image_tensor, tf.float32) / 255
        image_tensor = tf.expand_dims(image_tensor, axis=0)
        
        # 预测
        out = model(image_tensor)
        out = tf.argmax(out, axis=-1)
        
        # 获取文件名（保持原名称不变）
        image_name = os.path.basename(path_single)
        
        # 根据预测结果复制到相应目录
        if out.numpy()[0] == 1:
            dest_path = os.path.join(predict_True_path, image_name)
        else:
            dest_path = os.path.join(predict_False_path, image_name)
        
        # 复制文件（保持原文件名）
        shutil.copy2(path_single, dest_path)
        processed_count += 1
        
    except Exception as e:
        print(f"\n处理图片 {path_single} 时出错: {str(e)}")
        continue
    
    frame_time = time.time() - frame_start_time
    frame_times.append(frame_time)

# 性能统计
total_time = time.time() - total_start_time
avg_frame_time = sum(frame_times) / len(frame_times) if frame_times else 0
fps = len(frame_times) / total_time if total_time > 0 else 0

print("\n\n" + "="*50)
print("性能统计:")
print(f"总图片数: {len(fList)}")
print(f"成功处理: {processed_count}")
print(f"总耗时: {total_time:.2f} 秒")
print(f"平均每帧处理时间: {avg_frame_time:.4f} 秒")
print(f"处理速度: {fps:.2f} 帧/秒 (FPS)")
if frame_times:
    print(f"最快单帧: {min(frame_times):.4f} 秒")
    print(f"最慢单帧: {max(frame_times):.4f} 秒")
else:
    print("无有效处理数据")
print("="*50)
