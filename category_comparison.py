import matplotlib.pyplot as plt
import numpy as np

# === Collision probability results (Westbound) ===
results_without_vts = {
    "100-125m": 0.000500,
    "125-150m": 0.001500,
    "150-175m": 0.000500,
    "175-200m": 0.000500,
    "200-225m": 0.000500,
    "225-250m": 0.000500,
    "250-275m": 0.000500,
    "275-300m": 0.001000,
    "300-325m": 0.001500,
    "325-350m": 0.003000

   }

results_with_vts = {
"100-125m": 0.000075,
"125-150m": 0.000225,
"150-175m": 0.000075,
"175-200m": 0.000075,
"200-225m": 0.000075,
"225-250m": 0.000075,
"250-275m": 0.000075,
"275-300m": 0.000150,
"300-325m": 0.000225,
"325-350m": 0.000450,
}

# === Add total Pcat1 to the dictionaries ===
results_without_vts["Total"] = sum(results_without_vts.values())
results_with_vts["Total"] = sum(results_with_vts.values())

# === Convert to per million ===
scale = 1_000_000
classes = list(results_without_vts.keys())
values_no_vts = [results_without_vts[c] * scale for c in classes]
values_with_vts = [results_with_vts[c] * scale for c in classes]

x = np.arange(len(classes))
width = 0.35

# === Plotting ===
fig, ax = plt.subplots(figsize=(14, 6))
bars1 = ax.bar(x - width/2, values_no_vts, width, label='Without VTS/TTS', color='gray')
bars2 = ax.bar(x + width/2, values_with_vts, width, label='With VTS/TTS', color='green')

# === Annotate each bar ===
for bar in bars1 + bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.1f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=8)

# === Labels and formatting ===
ax.set_xlabel('Ship Length Class (LOA) + Total')
ax.set_ylabel('Collision Probability (per million)')
ax.set_title('Pcat1 Westbound Collision Probability per Ship Class\nWith vs Without VTS/TTS (including total)')
ax.set_xticks(x)
ax.set_xticklabels(classes, rotation=45)
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()
