import numpy as np
from scipy import stats
from scipy.odr import Model, RealData, ODR
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MultipleLocator, MaxNLocator

# 定义模型函数，横轴是高度，纵轴是半径
def model(beta, x):
    return beta[0] * x**beta[1]

# 读取数据，只读取特定列名的数据
def load_data(filename, col1, col2):
    df = pd.read_csv(filename)
    return df[[col1, col2]].values

# 指定列名
geo_R_col = 'geo_R'  # CSV文件中尘卷风半径列的实际列名
object_Height_col = 'object_Height'  # CSV文件中距离火星表面高度列的实际列名

# 读取数据
data = load_data("Analyse_marsyear36.csv", geo_R_col, object_Height_col)
geo_R = data[:, 0]
object_Height = data[:, 1]

# 创建模型和数据对象
linear = Model(model)
data = RealData(object_Height, geo_R, sx=100, sy=0.063)

# 创建正交距离回归对象并拟合
odr = ODR(data, linear, beta0=[1, 1])
out = odr.run()

# 打印拟合结果
beta = out.beta
print(out.beta)
print(out.sd_beta)

# 计算残差
fitted = model(beta, object_Height)
residuals = geo_R - fitted

# 计算相关系数、残差平方和、总平方和、决定系数
correlation = np.corrcoef(object_Height, geo_R)[0, 1]
print(f"Correlation coefficient: {correlation}")
ss_res = np.sum(residuals**2)
print(f"Residual sum of squares: {ss_res}")
ss_tot = np.sum((geo_R - np.mean(geo_R))**2)
print(f"Total sum of squares: {ss_tot}")
r_squared = 1 - (ss_res / ss_tot)
print(f"R-squared: {r_squared}")

# 绘制拟合曲线和数据点
plt.errorbar(object_Height, geo_R, xerr=100, yerr=0.063, fmt='o', label="Data")
plt.plot(np.linspace(0, max(object_Height), 100), model(beta, np.linspace(0, max(object_Height), 100)), label=f"Fit: R = {beta[0]:.2f}H^{beta[1]:.2f}")
plt.xlabel("Height (m)")
plt.ylabel("Dust Devil Radius (m)")
plt.legend()


# 设置横坐标轴的刻度
ax = plt.gca()

# 主要刻度，0-1000范围内每500一个间隔，超过1000的范围内每5000一个间隔
if max(object_Height) < 5000:
    ax.xaxis.set_major_locator(MultipleLocator(500))
else:
    ax.xaxis.set_major_locator(MultipleLocator(5000))

# 次要刻度，0-1000范围内每200一个间隔，超过1000的范围内每500一个间隔
ax.xaxis.set_minor_locator(MultipleLocator(200))

# 设置x轴范围以确保刻度正确显示
ax.set_xlim(0, max(object_Height))

# 显示图表
plt.show()




# 定义线性模型函数
def linear_model(beta, x):
    return beta[0] * x + beta[1]

# 创建模型和数据对象
linear_model_obj = Model(linear_model)
linear_data = RealData(object_Height, geo_R, sx=100, sy=0.063)

# 创建正交距离回归对象并拟合线性模型
linear_odr = ODR(linear_data, linear_model_obj, beta0=[1, 1])
linear_out = linear_odr.run()

# 打印线性拟合结果
linear_beta = linear_out.beta
print(linear_out.beta)
print(linear_out.sd_beta)

# 计算线性拟合的残差
linear_fitted = linear_model(linear_beta, object_Height)
linear_residuals = geo_R - linear_fitted

# 计算线性拟合的相关系数、残差平方和、总平方和、决定系数
linear_correlation = np.corrcoef(object_Height, geo_R)[0, 1]
print(f"Linear Correlation coefficient: {linear_correlation}")
linear_ss_res = np.sum(linear_residuals**2)
print(f"Linear Residual sum of squares: {linear_ss_res}")
linear_ss_tot = np.sum((geo_R - np.mean(geo_R))**2)
print(f"Linear Total sum of squares: {linear_ss_tot}")
linear_r_squared = 1 - (linear_ss_res / linear_ss_tot)
print(f"Linear R-squared: {linear_r_squared}")

# 绘制拟合曲线和数据点
plt.errorbar(object_Height, geo_R, xerr=100, yerr=0.063, fmt='o', label="Data")
plt.plot(np.linspace(0, max(object_Height), 100), model(beta, np.linspace(0, max(object_Height), 100)), label=f"Fit: R = {beta[0]:.2f}H^{beta[1]:.2f}")
plt.xlabel("Height (m)")
plt.ylabel("Dust Devil Radius (m)")
plt.legend()

# 设置横坐标轴的刻度
ax = plt.gca()
ax.xaxis.set_major_locator(MultipleLocator(100))  # 主要刻度
ax.xaxis.set_minor_locator(MultipleLocator(10))   # 次要刻度
ax.xaxis.set_major_locator(MaxNLocator(nbins=10, steps=[1, 3, 5, 10], min_n_ticks=3))  # 0-1000范围内大约10个刻度
ax.xaxis.set_minor_locator(MultipleLocator(50))   # 1000以上的次要刻度

# 显示图表
plt.show()

# 对数据进行对数转换
log_object_Height = np.log(object_Height)
log_geo_R = np.log(geo_R)

# 确保没有负数或零，因为对数函数在这些值上未定义
valid_indices = (object_Height > 0) & (geo_R > 0)
log_object_Height = log_object_Height[valid_indices]
log_geo_R = log_geo_R[valid_indices]

# 创建对数转换后的数据对象
log_data = RealData(log_object_Height, log_geo_R, sx=1, sy=1)

# 使用线性模型拟合对数转换后的数据
log_linear_odr = ODR(log_data, linear_model_obj, beta0=[1, 1])
log_linear_out = log_linear_odr.run()

# 打印对数转换后线性拟合的结果
log_linear_beta = log_linear_out.beta
print(log_linear_out.beta)
print(log_linear_out.sd_beta)

# 计算对数转换后线性拟合的残差
log_linear_fitted = linear_model(log_linear_beta, log_object_Height)
log_linear_residuals = log_geo_R - log_linear_fitted

# 计算对数转换后线性拟合的相关系数、残差平方和、总平方和、决定系数
log_linear_correlation = np.corrcoef(log_object_Height, log_geo_R)[0, 1]
print(f"Log-Linear Correlation coefficient: {log_linear_correlation}")
log_linear_ss_res = np.sum(log_linear_residuals**2)
print(f"Log-Linear Residual sum of squares: {log_linear_ss_res}")
log_linear_ss_tot = np.sum((log_geo_R - np.mean(log_geo_R))**2)
print(f"Log-Linear Total sum of squares: {log_linear_ss_tot}")
log_linear_r_squared = 1 - (log_linear_ss_res / log_linear_ss_tot)
print(f"Log-Linear R-squared: {log_linear_r_squared}")

# 绘制对数转换后的数据和对数线性拟合曲线
plt.scatter(log_object_Height, log_geo_R, label="Log-Transformed Data")
plt.plot(np.linspace(min(log_object_Height), max(log_object_Height), 100), 
         linear_model(log_linear_beta, np.linspace(min(log_object_Height), max(log_object_Height), 100)), 
         label=f"Log-Linear Fit: ln(R) = {log_linear_beta[0]:.2f}ln(H) + {log_linear_beta[1]:.2f}")
plt.xlabel("ln(Height) (m)")
plt.ylabel("ln(Dust Devil Radius) (m)")
plt.legend()
plt.title("Log-Linear Fit of ln(Height) vs ln(Dust Devil Radius)")
plt.show()
