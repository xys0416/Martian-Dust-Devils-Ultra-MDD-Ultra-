import pandas as pd
import numpy as np
import math

# 读取CSV文件
filepath = 'Analyse_marsyear36.csv'
df = pd.read_csv(filepath)

# 计算太阳赤纬 δs
def calculate_solar_declination(Ls):
    return math.asin(0.4256 * math.sin(math.radians(Ls))) + 0.25 * math.sin(math.radians(Ls))

# 计算太阳时角 H
def calculate_hour_angle(LTST):
    return 15 * (LTST - 12)

# 计算太阳高度角 h
def calculate_solar_elevation(Ls, LTST, latitude):
    δs = calculate_solar_declination(Ls)
    H = calculate_hour_angle(LTST)
    return math.degrees(math.asin(math.sin(math.radians(latitude)) * math.sin(δs) + 
                                  math.cos(math.radians(latitude)) * math.cos(δs) * math.cos(math.radians(H))))

# 应用改进后的函数计算太阳高度角并保存到新列
df['Solar_altitude'] = df.apply(lambda row: calculate_solar_elevation(row['Ls'], row['LTST'], row['latitude']), axis=1)

# 保存到 CSV 文件
df.to_csv('updated_' + filepath, index=False)

# 定义一个函数来计算物体的高度
def calculate_object_height(row):
    # 确保geo_A列中的值是浮点数，如果不是，则返回NaN
    geo_A = row['geo_A']
    
    # 计算太阳高度角的正切值
    tan_solar_altitude = math.tan(math.radians(row['Solar_altitude']))
    
    # 计算物体高度
    object_height = geo_A * tan_solar_altitude
    return object_height

# 选择Type_列为'shadow'的行
shadow_rows = df[df['Type'] == 'shadow']

# 计算这些行的物体高度
shadow_rows['object_Height'] = shadow_rows.apply(calculate_object_height, axis=1)

# 将计算出的物体高度更新到整个DataFrame中
df.loc[shadow_rows.index, 'object_Height'] = shadow_rows['object_Height']

# 保存到CSV文件
df.to_csv(filepath, index=False)

