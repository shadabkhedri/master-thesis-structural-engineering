import pandas as pd
import numpy as np
import os
import glob
from shapely.geometry import LineString
from pyproj import Transformer
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# === Configuration ===
folder_path = r""
excluded_mmsis = {41536300, 41536301, 257017920, 257017930, 258573000, 257268700}
bin_size = 1.0  # meters

# === Anda and Lote coordinates in (latitude, longitude) ===
anda_lat, anda_lon = 61.84645, 6.08237
lote_lat, lote_lon = 61.86406, 6.07931

# === Coordinate transformation: WGS84 to UTM Zone 32N (accurate for Norway)
transformer = Transformer.from_crs("EPSG:4326", "EPSG:32632", always_xy=True)
anda_proj = transformer.transform(anda_lon, anda_lat)
lote_proj = transformer.transform(lote_lon, lote_lat)

# === Bridge axis and length ===
bridge_line = LineString([anda_proj, lote_proj])
bridge_length = bridge_line.length
print(f"Projected bridge length: {bridge_length:.2f} meters")

# === Load and clean AIS data ===
files = glob.glob(os.path.join(folder_path, "*.csv"))
df_list = []

for file in files:
    try:
        df = pd.read_csv(file, delimiter="\t", encoding="utf-8", on_bad_lines='skip')
        df.columns = [col.lower().strip() for col in df.columns]

        required = ['mmsi', 'latitude', 'longitude', 'date_time_utc']
        if all(col in df.columns for col in required):
            df = df.dropna(subset=required)
            df['mmsi'] = pd.to_numeric(df['mmsi'], errors='coerce')
            df = df[~df['mmsi'].isin(excluded_mmsis)]
            df['date_time_utc'] = pd.to_datetime(df['date_time_utc'], errors='coerce')
            df = df.sort_values(by=['mmsi', 'date_time_utc'])
            df_list.append(df)
        else:
            print(f"Skipping {file} — missing required columns.")
    except Exception as e:
        print(f"Error reading {file}: {e}")

# === Combine all cleaned data ===
df = pd.concat(df_list, ignore_index=True)

# === Generate ship LineStrings and separate directions ===
east_crossings = []
west_crossings = []
bridge_vector = np.array([lote_proj[0] - anda_proj[0], lote_proj[1] - anda_proj[1]])

for mmsi, group in df.groupby('mmsi'):
    if len(group) < 2:
        continue
    coords = [transformer.transform(lon, lat) for lon, lat in zip(group['longitude'], group['latitude'])]
    ship_line = LineString(coords)
    if ship_line.crosses(bridge_line) or ship_line.intersects(bridge_line):
        intersection = ship_line.intersection(bridge_line)
        if intersection.is_empty:
            continue
        # Direction detection
        vec = np.array(coords[-1]) - np.array(coords[0])
        direction = np.dot(vec, bridge_vector)
        if intersection.geom_type == 'Point':
            dist = bridge_line.project(intersection)
            if direction > 0:
                east_crossings.append(dist)
            else:
                west_crossings.append(dist)
        elif intersection.geom_type == 'MultiPoint':
            for pt in intersection.geoms:
                dist = bridge_line.project(pt)
                if direction > 0:
                    east_crossings.append(dist)
                else:
                    west_crossings.append(dist)

# === Plot KDE for east and westbound separately ===
x_vals = np.linspace(0, bridge_length, 500)

# KDEs
kde_east = gaussian_kde(east_crossings, bw_method=0.05) if east_crossings else None
kde_west = gaussian_kde(west_crossings, bw_method=0.05) if west_crossings else None
kde_east_vals = kde_east(x_vals) * len(east_crossings) * bin_size if kde_east else np.zeros_like(x_vals)
kde_west_vals = kde_west(x_vals) * len(west_crossings) * bin_size if kde_west else np.zeros_like(x_vals)

# Plot
plt.figure(figsize=(10, 5))
if east_crossings:
    plt.plot(x_vals, kde_east_vals, label='Eastbound KDE', color='blue')
if west_crossings:
    plt.plot(x_vals, kde_west_vals, label='Westbound KDE', color='red')

plt.xlabel("Distance along bridge axis (m)")
plt.ylabel("Estimated ship crossing density")
plt.title("Smoothed Ship Crossing Density (Anda–Lote)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
