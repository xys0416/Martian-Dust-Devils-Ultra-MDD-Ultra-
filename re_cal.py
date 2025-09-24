import pandas as pd
from PIL import Image

# 读取CSV文件
csv_file = 'coordinate_targets_180_270.csv'  # 请替换为你的CSV文件路径
df = pd.read_csv(csv_file)

# 图片文件夹路径
image_folder = '270TRUE'  # 请替换为你的图片文件夹路径


# 已知每个像素的地理大小
pixel_size = 5.86874

# 创建两个新列用于存储左上角坐标，并初始化为NaN
df['left_upper_x'] = pd.NA
df['left_upper_y'] = pd.NA

# 遍历CSV中的每一行
for index, row in df.iterrows():
    # 图片完整路径
    image_path = f"{image_folder}/{row['coordinate_name']}"
    
    # 获取图片尺寸
    try:
        with Image.open(image_path) as img:
            width, height = img.size
    except FileNotFoundError:
        print(f"Image not found: {image_path}")
        continue
    
    # 计算左上角坐标
    left_upper_x = row['geographic_x'] - (width * pixel_size / 2)
    left_upper_y = row['geographic_y'] + (height * pixel_size / 2)
    
    # 更新DataFrame
    df.at[index, 'left_upper_x'] = left_upper_x
    df.at[index, 'left_upper_y'] = left_upper_y

# 将更新后的DataFrame保存回原始CSV文件
df.to_csv(csv_file, index=False)

