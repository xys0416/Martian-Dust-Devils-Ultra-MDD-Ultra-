#This document calculates the solar altitude and altitude of the dust devil.
import pandas as pd
import numpy as np
import math

# ------------------------------------------------------------------
# 1. Read CSV file
# ------------------------------------------------------------------
filepath = 'Analyse_marsyear36.csv'
df = pd.read_csv(filepath)

# ------------------------------------------------------------------
# 2. Solar‐geometry helper functions
# ------------------------------------------------------------------
def calculate_solar_declination(Ls: float) -> float:
    """
    Compute Mars solar declination δs (radians) from areocentric longitude Ls (degrees).
    """
    return math.asin(0.4256 * math.sin(math.radians(Ls))) + 0.25 * math.sin(math.radians(Ls))

def calculate_hour_angle(LTST: float) -> float:
    """
    Compute solar hour angle H (degrees) from Local True Solar Time LTST (hours).
    """
    return 15 * (LTST - 12)

def calculate_solar_elevation(Ls: float, LTST: float, latitude: float) -> float:
    """
    Compute solar elevation angle (degrees) above horizon.
    """
    δs  = calculate_solar_declination(Ls)
    H   = calculate_hour_angle(LTST)
    lat = math.radians(latitude)

    sin_el = math.sin(lat) * math.sin(δs) + \
             math.cos(lat) * math.cos(δs) * math.cos(math.radians(H))
    return math.degrees(math.asin(sin_el))

# ------------------------------------------------------------------
# 3. Add solar altitude column
# ------------------------------------------------------------------
df['Solar_altitude'] = df.apply(
    lambda row: calculate_solar_elevation(row['Ls'], row['LTST'], row['latitude']),
    axis=1
)

# ------------------------------------------------------------------
# 4. Save intermediate result (optional)
# ------------------------------------------------------------------
df.to_csv('updated_' + filepath, index=False)

# ------------------------------------------------------------------
# 5. Compute object height from shadow length
# ------------------------------------------------------------------
def calculate_object_height(row: pd.Series) -> float:
    """
    Estimate object height using shadow length (geo_A) and solar altitude.
    Returns NaN if geo_A is missing or non-numeric.
    """
    try:
        geo_A = float(row['geo_A'])
    except (ValueError, TypeError):
        return np.nan

    tan_solar_alt = math.tan(math.radians(row['Solar_altitude']))
    return geo_A * tan_solar_alt

# Apply height calculation only to rows marked as 'shadow'
shadow_mask = df['Type'] == 'shadow'
df.loc[shadow_mask, 'object_Height'] = df[shadow_mask].apply(calculate_object_height, axis=1)

# ------------------------------------------------------------------
# 6. Final save (overwrites original file)
# ------------------------------------------------------------------
df.to_csv(filepath, index=False)
print("Processing complete – file saved to", filepath)
