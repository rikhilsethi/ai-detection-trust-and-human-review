import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load our predictions
df = pd.read_csv('predictions.csv')

print(f"Total predictions: {len(df)}")
print(f"Overall accuracy: {df['correct'].mean():.3f}")
print(f"Mean confidence: {df['confidence'].mean():.3f}")

# --- RELIABILITY DIAGRAM ---
# Split predictions into 10 bins by confidence score
# For each bin: what fraction were actually correct?

n_bins = 10
bin_edges = np.linspace(0, 1, n_bins + 1)

bin_accuracy = []
bin_confidence = []
bin_counts = []

for i in range(n_bins):
    low = bin_edges[i]
    high = bin_edges[i+1]
    
    # Get all predictions in this confidence range
    mask = (df['confidence'] >= low) & (df['confidence'] < high)
    bin_df = df[mask]
    
    if len(bin_df) > 0:
        avg_confidence = bin_df['confidence'].mean()
        avg_accuracy = bin_df['correct'].mean()
        bin_accuracy.append(avg_accuracy)
        bin_confidence.append(avg_confidence)
        bin_counts.append(len(bin_df))
    else:
        bin_accuracy.append(0)
        bin_confidence.append((low + high) / 2)
        bin_counts.append(0)

# Plot reliability diagram
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Perfect calibration line
ax1.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=2)

# Actual calibration
ax1.bar(bin_confidence, bin_accuracy, width=0.08, alpha=0.7, 
        color='steelblue', label='Model Calibration', edgecolor='black')

ax1.set_xlabel('Confidence Score', fontsize=12)
ax1.set_ylabel('Actual Accuracy', fontsize=12)
ax1.set_title('Reliability Diagram\n(How trustworthy is the confidence score?)', fontsize=13)
ax1.legend()
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.grid(True, alpha=0.3)

# Add count labels
for i, (conf, acc, count) in enumerate(zip(bin_confidence, bin_accuracy, bin_counts)):
    if count > 0:
        ax1.text(conf, acc + 0.02, f'n={count}', ha='center', fontsize=8)

# --- EXPECTED CALIBRATION ERROR ---
# ECE = weighted average of |confidence - accuracy| across bins
ece = 0
total = len(df)
for i in range(n_bins):
    if bin_counts[i] > 0:
        weight = bin_counts[i] / total
        ece += weight * abs(bin_confidence[i] - bin_accuracy[i])

print(f"\nExpected Calibration Error (ECE): {ece:.4f}")
print(f"(0 = perfectly calibrated, 1 = completely wrong)")

# Confidence distribution
ax2.hist(df['confidence'], bins=20, color='steelblue', 
         edgecolor='black', alpha=0.7)
ax2.set_xlabel('Confidence Score', fontsize=12)
ax2.set_ylabel('Number of Detections', fontsize=12)
ax2.set_title('Distribution of Confidence Scores', fontsize=13)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('calibration_results.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nPlot saved as calibration_results.png")

# Print bin by bin breakdown
print("\n--- BIN BY BIN BREAKDOWN ---")
print(f"{'Confidence Range':<20} {'Avg Confidence':<18} {'Actual Accuracy':<18} {'Count':<8} {'Gap'}")
print("-" * 80)
for i in range(n_bins):
    low = bin_edges[i]
    high = bin_edges[i+1]
    if bin_counts[i] > 0:
        gap = bin_confidence[i] - bin_accuracy[i]
        print(f"{low:.1f} - {high:.1f}{'':12} {bin_confidence[i]:.3f}{'':12} {bin_accuracy[i]:.3f}{'':12} {bin_counts[i]:<8} {gap:+.3f}")