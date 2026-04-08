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
anda_proj = transformer.transform(anda_lon, anda_lat)  # (lon, lat)
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

# === Generate ship LineStrings ===
ship_lines = []
for mmsi, group in df.groupby('mmsi'):
    if len(group) < 2:
        continue
    coords = [transformer.transform(lon, lat) for lon, lat in zip(group['longitude'], group['latitude'])]
    ship_lines.append(LineString(coords))

# === Project all crossing distances for KDE and histogram ===
crossing_distances = []

for ship in ship_lines:
    if ship.crosses(bridge_line) or ship.intersects(bridge_line):
        intersection = ship.intersection(bridge_line)
        if intersection.geom_type == 'Point':
            dist = bridge_line.project(intersection)
            crossing_distances.append(dist)
        elif intersection.geom_type == 'MultiPoint':
            for pt in intersection.geoms:
                dist = bridge_line.project(pt)
                crossing_distances.append(dist)

# === Histogram (binned counts) ===
bin_edges = np.arange(0, bridge_length + bin_size, bin_size)
hist_counts, _ = np.histogram(crossing_distances, bins=bin_edges)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# === KDE ===
kde = gaussian_kde(crossing_distances, bw_method=0.05)
x_vals = np.linspace(0, bridge_length, 500)
kde_vals = kde(x_vals)

# === Plot both ===
plt.figure(figsize=(10, 5))
plt.bar(bin_centers, hist_counts, width=bin_size, alpha=0.4, label='Raw Histogram', color='gray', edgecolor='black')
plt.plot(x_vals, kde_vals * len(crossing_distances) * bin_size, color='blue', label='Smoothed KDE')

plt.xlabel("Distance along bridge axis (m)")
plt.ylabel("Ship crossings (count & estimated density)")
plt.title("Ship Crossings Along Bridge Axis (Anda–Lote)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
