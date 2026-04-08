import os
import pandas as pd
import numpy as np
from shapely.geometry import LineString, Point
from pyproj import Transformer
import matplotlib.pyplot as plt
import glob

def estimate_crossing_speeds_from_directory(
    folder_path,
    anda_latlon=(61.84645, 6.08237),
    lote_latlon=(61.86406, 6.07931),
    excluded_mmsis={41536300, 41536301, 257017920, 257017930, 258573000, 257268700}
):
    transformer = Transformer.from_crs("epsg:4326", "epsg:3857", always_xy=True)
    anda_proj = transformer.transform(anda_latlon[1], anda_latlon[0])
    lote_proj = transformer.transform(lote_latlon[1], lote_latlon[0])
    bridge_line = LineString([anda_proj, lote_proj])
    speeds_at_crossings = []

    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

    for file in csv_files:
        try:
            df = pd.read_csv(file, delimiter="\t", encoding="utf-8", on_bad_lines='skip')
            df.columns = [col.lower().strip() for col in df.columns]

            if not all(col in df.columns for col in ['mmsi', 'latitude', 'longitude', 'date_time_utc']):
                continue

            df = df.dropna(subset=['mmsi', 'latitude', 'longitude', 'date_time_utc'])
            df['mmsi'] = pd.to_numeric(df['mmsi'], errors='coerce')
            df = df[~df['mmsi'].isin(excluded_mmsis)]
            df['date_time_utc'] = pd.to_datetime(df['date_time_utc'], errors='coerce')

            for mmsi, group in df.groupby('mmsi'):
                if len(group) < 2:
                    continue
                group = group.sort_values('date_time_utc')
                coords = [transformer.transform(lon, lat) for lon, lat in zip(group['longitude'], group['latitude'])]
                sogs = group['speed_over_ground'].values if 'speed_over_ground' in group.columns else None
                if sogs is None:
                    continue

                ship_line = LineString(coords)
                if ship_line.crosses(bridge_line):
                    intersection = ship_line.intersection(bridge_line)
                    if intersection.geom_type == 'Point':
                        dists = [Point(c).distance(intersection) for c in coords]
                        min_index = int(np.argmin(dists))
                        sog_knots = sogs[min_index]
                        if pd.notnull(sog_knots):
                            sog_mps = sog_knots * 0.5144
                            speeds_at_crossings.append(sog_mps)

        except Exception as e:
            print(f"Error processing {file}: {e}")

    return speeds_at_crossings


# === Usage example ===
ais_folder = r"C:\Users\47463\OneDrive\Attachments\Desktop\python_tutorial\Nordfjorden 10-2023 - 10-2024"
speeds = estimate_crossing_speeds_from_directory(ais_folder)

# === Output statistics ===
if speeds:
    print(f"Average speed at crossing: {np.mean(speeds):.2f} m/s")
    print(f"Standard deviation: {np.std(speeds):.2f} m/s")

    # Plot histogram
    plt.hist(speeds, bins=20, color='skyblue', edgecolor='black')
    plt.xlabel("Impact velocity (m/s)")
    plt.ylabel("Number of ships")
    plt.title("Ship Speeds at Anda–Lote Bridge Crossing")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
else:
    print("No crossing speeds were detected.")
