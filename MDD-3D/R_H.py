#The fitting results for radius and height.
import numpy as np
from scipy import stats
from scipy.odr import Model, RealData, ODR
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MultipleLocator, MaxNLocator

# ------------------------------------------------------------------
# 1. Power-law model  R = β0 * H^β1
# ------------------------------------------------------------------
def model(beta, x):
    """
    Power-law relationship: Radius = beta[0] * Height^beta[1]
    x : independent variable (Height)
    beta[0] : prefactor
    beta[1] : exponent
    """
    return beta[0] * x**beta[1]

# ------------------------------------------------------------------
# 2. Load data
# ------------------------------------------------------------------
def load_data(filename, col1, col2):
    """Read two specified columns from CSV."""
    df = pd.read_csv(filename)
    return df[[col1, col2]].values

geo_R_col         = 'geo_R'          # column name for dust-devil radius
object_Height_col = 'object_Height'  # column name for object height above surface

data          = load_data("Analyse_marsyear36.csv", geo_R_col, object_Height_col)
geo_R         = data[:, 0]
object_Height = data[:, 1]

# ------------------------------------------------------------------
# 3. Orthogonal Distance Regression (ODR) for power-law
# ------------------------------------------------------------------
linear = Model(model)
data_obj = RealData(object_Height, geo_R, sx=100, sy=0.063)  # x & y uncertainties

odr  = ODR(data_obj, linear, beta0=[1, 1])
out  = odr.run()

beta     = out.beta        # fitted parameters
beta_err = out.sd_beta     # standard errors

print("Power-law fit  R = β0·H^β1")
print(f"β0 = {beta[0]:.4f} ± {beta_err[0]:.4f}")
print(f"β1 = {beta[1]:.4f} ± {beta_err[1]:.4f}")

# ------------------------------------------------------------------
# 4. Goodness-of-fit statistics
# ------------------------------------------------------------------
fitted    = model(beta, object_Height)
residuals = geo_R - fitted

correlation = np.corrcoef(object_Height, geo_R)[0, 1]
ss_res      = np.sum(residuals**2)
ss_tot      = np.sum((geo_R - np.mean(geo_R))**2)
r_squared   = 1 - (ss_res / ss_tot)

print(f"Correlation coefficient : {correlation:.4f}")
print(f"Residual sum of squares : {ss_res:.4g}")
print(f"Total sum of squares    : {ss_tot:.4g}")
print(f"R-squared               : {r_squared:.4f}")

# ------------------------------------------------------------------
# 5. Plot power-law fit
# ------------------------------------------------------------------
plt.errorbar(object_Height, geo_R, xerr=100, yerr=0.063,
             fmt='o', label="Data")
x_smooth = np.linspace(0, max(object_Height), 100)
plt.plot(x_smooth, model(beta, x_smooth),
         label=f"Fit:  R = {beta[0]:.2f}·H^{beta[1]:.2f}")

plt.xlabel("Height (m)")
plt.ylabel("Dust Devil Radius (m)")
plt.legend()

# Major/minor ticks
ax = plt.gca()
if max(object_Height) < 5000:
    ax.xaxis.set_major_locator(MultipleLocator(500))
    ax.xaxis.set_minor_locator(MultipleLocator(200))
else:
    ax.xaxis.set_major_locator(MultipleLocator(5000))
    ax.xaxis.set_minor_locator(MultipleLocator(500))
ax.set_xlim(0, max(object_Height))

plt.show()

# ------------------------------------------------------------------
# 6. Linear ODR  R = β0·H + β1  (for comparison)
# ------------------------------------------------------------------
def linear_model(beta, x):
    return beta[0] * x + beta[1]

linear_data_obj = RealData(object_Height, geo_R, sx=100, sy=0.063)
linear_odr      = ODR(linear_data_obj, Model(linear_model), beta0=[1, 1])
linear_out      = linear_odr.run()

linear_beta     = linear_out.beta
linear_beta_err = linear_out.sd_beta

print("\nLinear fit  R = β0·H + β1")
print(f"β0 = {linear_beta[0]:.4f} ± {linear_beta_err[0]:.4f}")
print(f"β1 = {linear_beta[1]:.4f} ± {linear_beta_err[1]:.4f}")

# Statistics
linear_fitted    = linear_model(linear_beta, object_Height)
linear_residuals = geo_R - linear_fitted
linear_ss_res    = np.sum(linear_residuals**2)
linear_r_squared = 1 - (linear_ss_res / ss_tot)  # same ss_tot as before

print(f"Linear R-squared : {linear_r_squared:.4f}")

# Quick plot
plt.errorbar(object_Height, geo_R, xerr=100, yerr=0.063,
             fmt='o', label="Data")
plt.plot(x_smooth, linear_model(linear_beta, x_smooth),
         label=f"Linear: R = {linear_beta[0]:.2f}·H + {linear_beta[1]:.2f}")
plt.xlabel("Height (m)")
plt.ylabel("Dust Devil Radius (m)")
plt.legend()
ax = plt.gca()
ax.xaxis.set_major_locator(MultipleLocator(100))
ax.xaxis.set_minor_locator(MultipleLocator(10))
plt.show()

# ------------------------------------------------------------------
# 7. Log-log linear fit  ln(R) = β0·ln(H) + β1
# ------------------------------------------------------------------
# Remove non-positive values before taking log
valid_idx   = (object_Height > 0) & (geo_R > 0)
log_H       = np.log(object_Height[valid_idx])
log_R       = np.log(geo_R[valid_idx])

log_data_obj = RealData(log_H, log_R, sx=1, sy=1)
log_odr      = ODR(log_data_obj, Model(linear_model), beta0=[1, 1])
log_out      = log_odr.run()

log_beta     = log_out.beta
log_beta_err = log_out.sd_beta

print("\nLog-log fit  ln(R) = β0·ln(H) + β1")
print(f"β0 = {log_beta[0]:.4f} ± {log_beta_err[0]:.4f}")
print(f"β1 = {log_beta[1]:.4f} ± {log_beta_err[1]:.4f}")

# Statistics in log space
log_fitted    = linear_model(log_beta, log_H)
log_residuals = log_R - log_fitted
log_ss_res    = np.sum(log_residuals**2)
log_ss_tot    = np.sum((log_R - np.mean(log_R))**2)
log_r_squared = 1 - (log_ss_res / log_ss_tot)

print(f"Log-log R-squared : {log_r_squared:.4f}")

# Plot log-log
plt.scatter(log_H, log_R, label="Log-transformed data")
x_log = np.linspace(min(log_H), max(log_H), 100)
plt.plot(x_log, linear_model(log_beta, x_log),
         label=f"Fit: ln(R) = {log_beta[0]:.2f}·ln(H) + {log_beta[1]:.2f}")
plt.xlabel("ln(Height) (m)")
plt.ylabel("ln(Dust Devil Radius) (m)")
plt.title("Log-linear fit of ln(Height) vs ln(Dust Devil Radius)")
plt.legend()
plt.show()
