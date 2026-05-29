import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load predictions
df = pd.read_csv('predictions.csv')

# --- COST FUNCTION ---
# In a safety-critical system:
# Missing a real threat (false negative) is more costly than a false alarm (false positive)
# We can adjust these weights to reflect different risk tolerances

FP_COST = 1   # Cost of flagging something incorrectly
FN_COST = 5   # Cost of missing a real threat (5x more costly)

# Test every possible threshold from 0.1 to 0.99
thresholds = np.linspace(0.1, 0.99, 200)

total_costs = []
fp_rates = []
fn_rates = []
precision_list = []
recall_list = []

for threshold in thresholds:
    # Everything above threshold gets flagged for review
    flagged = df['confidence'] >= threshold
    not_flagged = ~flagged

    # True positives: correctly flagged real threats
    tp = ((flagged) & (df['correct'] == 1)).sum()
    
    # False positives: flagged but actually wrong
    fp = ((flagged) & (df['correct'] == 0)).sum()
    
    # False negatives: not flagged but should have been
    fn = ((not_flagged) & (df['correct'] == 1)).sum()
    
    # True negatives: correctly not flagged
    tn = ((not_flagged) & (df['correct'] == 0)).sum()

    # Calculate cost
    total_cost = (fp * FP_COST) + (fn * FN_COST)
    total_costs.append(total_cost)

    # Calculate rates
    fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    fn_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    fp_rates.append(fp_rate)
    fn_rates.append(fn_rate)
    precision_list.append(precision)
    recall_list.append(recall)

# Find optimal threshold
optimal_idx = np.argmin(total_costs)
optimal_threshold = thresholds[optimal_idx]
optimal_cost = total_costs[optimal_idx]

print(f"Optimal Threshold: {optimal_threshold:.3f}")
print(f"Minimum Total Cost: {optimal_cost:.0f}")
print(f"At optimal threshold:")
print(f"  False Positive Rate: {fp_rates[optimal_idx]:.3f}")
print(f"  False Negative Rate: {fn_rates[optimal_idx]:.3f}")
print(f"  Precision: {precision_list[optimal_idx]:.3f}")
print(f"  Recall: {recall_list[optimal_idx]:.3f}")

# --- PLOTTING ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Total cost vs threshold
axes[0,0].plot(thresholds, total_costs, color='darkred', linewidth=2)
axes[0,0].axvline(x=optimal_threshold, color='green', linestyle='--', 
                   linewidth=2, label=f'Optimal: {optimal_threshold:.3f}')
axes[0,0].set_xlabel('Threshold', fontsize=11)
axes[0,0].set_ylabel('Total Cost', fontsize=11)
axes[0,0].set_title('Total Cost vs Threshold\n(FP cost=1, FN cost=5)', fontsize=12)
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# Plot 2: FP and FN rates vs threshold
axes[0,1].plot(thresholds, fp_rates, color='orange', linewidth=2, label='False Positive Rate')
axes[0,1].plot(thresholds, fn_rates, color='red', linewidth=2, label='False Negative Rate')
axes[0,1].axvline(x=optimal_threshold, color='green', linestyle='--',
                   linewidth=2, label=f'Optimal: {optimal_threshold:.3f}')
axes[0,1].set_xlabel('Threshold', fontsize=11)
axes[0,1].set_ylabel('Rate', fontsize=11)
axes[0,1].set_title('False Positive & False Negative Rates\nvs Threshold', fontsize=12)
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.3)

# Plot 3: Precision vs Recall
axes[1,0].plot(recall_list, precision_list, color='steelblue', linewidth=2)
axes[1,0].scatter([recall_list[optimal_idx]], [precision_list[optimal_idx]], 
                   color='green', s=100, zorder=5, label=f'Optimal threshold')
axes[1,0].set_xlabel('Recall', fontsize=11)
axes[1,0].set_ylabel('Precision', fontsize=11)
axes[1,0].set_title('Precision-Recall Curve', fontsize=12)
axes[1,0].legend()
axes[1,0].grid(True, alpha=0.3)

# Plot 4: Cost sensitivity - what if FN cost changes?
fn_costs = [1, 2, 5, 10, 20]
colors = ['blue', 'green', 'red', 'purple', 'orange']

for fn_cost, color in zip(fn_costs, colors):
    costs = []
    for i, threshold in enumerate(thresholds):
        flagged = df['confidence'] >= threshold
        not_flagged = ~flagged
        fp = ((flagged) & (df['correct'] == 0)).sum()
        fn = ((not_flagged) & (df['correct'] == 1)).sum()
        costs.append(fp * 1 + fn * fn_cost)
    
    opt_idx = np.argmin(costs)
    axes[1,1].plot(thresholds, costs, color=color, linewidth=1.5, 
                   label=f'FN cost={fn_cost} (opt={thresholds[opt_idx]:.2f})')

axes[1,1].set_xlabel('Threshold', fontsize=11)
axes[1,1].set_ylabel('Total Cost', fontsize=11)
axes[1,1].set_title('Cost Sensitivity Analysis\n(How does optimal threshold change with FN cost?)', fontsize=12)
axes[1,1].legend(fontsize=8)
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('threshold_optimization.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nPlot saved as threshold_optimization.png")