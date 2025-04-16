import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Parameters
# ----------------------------
gamma = 0.8
k = 4
x = 14

# Compute the ratio bounds using the given formulas:
# Lower bound: gamma^(k + x)
# Upper bound: gamma^(k - 1)
r_low = gamma ** (k + x)
r_high = gamma ** (k - 1)

# ----------------------------
# Configuration for plotting mode
# ----------------------------
# Options for plot_mode:
#   'w0'   -> Only fixed w[0] (default)
#   'w1'   -> Only fixed w[1]
#   'both' -> Both fixed w[0] and fixed w[1] subplots (side-by-side)
plot_mode = 'w0'  # change to 'w1' or 'both' as desired

# ----------------------------
# CASE 1: Fixed w[0] on x-axis; compute allowed region for w[1]
# ----------------------------
w0 = np.arange(0.01, 1.01, 0.01)

# Candidate bounds for w[1]:
# Lower candidate: r_low * w[0] = gamma^(k+x) * w[0]
# Upper candidate: r_high * w[0] = gamma^(k-1) * w[0]
cand_lower_w1 = r_low * w0
cand_upper_w1 = r_high * w0

# Apply sum constraint: w[0] + w[1] <= 1, so effective upper bound for w[1] is min(candidate, 1 - w[0])
eff_upper_w1 = np.minimum(cand_upper_w1, 1 - w0)
# If candidate lower bound is too high (i.e. w[0] + candidate lower > 1), mark these as invalid (NaN)
valid_mask_w1 = cand_lower_w1 <= (1 - w0)
eff_lower_w1 = np.where(valid_mask_w1, cand_lower_w1, np.nan)
eff_upper_w1 = np.where(valid_mask_w1, eff_upper_w1, np.nan)

# ----------------------------
# CASE 2: Fixed w[1] on x-axis; compute allowed region for w[0]
# ----------------------------
w1 = np.arange(0.01, 1.01, 0.01)

# Rearranged candidate bounds for w[0]:
# Lower candidate: w[1]/r_high = w[1]/(gamma^(k-1))
# Upper candidate: w[1]/r_low = w[1]/(gamma^(k+x))
cand_lower_w0 = w1 / r_high
cand_upper_w0 = w1 / r_low

# Sum constraint: w[0] <= 1 - w[1]
eff_upper_w0 = np.minimum(cand_upper_w0, 1 - w1)
valid_mask_w0 = cand_lower_w0 <= (1 - w1)
eff_lower_w0 = np.where(valid_mask_w0, cand_lower_w0, np.nan)
eff_upper_w0 = np.where(valid_mask_w0, eff_upper_w0, np.nan)

# ----------------------------
# PLOTTING
# ----------------------------
if plot_mode == 'w0':
    # Plot only for fixed w[0]
    plt.figure(figsize=(8, 6))
    plt.fill_between(w0, eff_lower_w1, eff_upper_w1, color='lightblue', alpha=0.5,
                     label='Allowed region for w[1]')
    plt.plot(w0, cand_lower_w1, 'b--', label=f'Candidate lower: {gamma}^({k}+{x}) = {r_low:.3f} * w[0]')
    plt.plot(w0, cand_upper_w1, 'r--', label=f'Candidate upper: {gamma}^({k - 1}) = {r_high:.3f} * w[0]')
    plt.plot(w0, 1 - w0, 'k-', label='Sum constraint: w[0] + w[1] = 1')
    plt.xlabel('Fixed weight w[0]')
    plt.ylabel('Computed weight w[1]')
    plt.title(f"Allowed w[1] vs. w[0] with Ratio & Sum Constraints, k={k} & x={x} & γ={gamma}")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('plot_fixed_w0.png', dpi=300)
    plt.show()
    print("Plot for fixed w[0] has been saved as 'plot_fixed_w0.png'.")

elif plot_mode == 'w1':
    # Plot only for fixed w[1]
    plt.figure(figsize=(8, 6))
    plt.fill_between(w1, eff_lower_w0, eff_upper_w0, color='lightgreen', alpha=0.5,
                     label='Allowed region for w[0]')
    plt.plot(w1, cand_lower_w0, 'g--', label=f'Candidate lower: w[1]/{gamma}^({k - 1}) = {cand_lower_w0[0]:.3f} (scaled)')
    plt.plot(w1, cand_upper_w0, 'm--', label=f'Candidate upper: w[1]/{gamma}^({k}+{x}) = {cand_upper_w0[0]:.3f} (scaled)')
    plt.plot(w1, 1 - w1, 'k-', label='Sum constraint: w[0] + w[1] = 1')
    plt.xlabel('Fixed weight w[1]')
    plt.ylabel('Computed weight w[0]')
    plt.title(f"Allowed w[1] vs. w[0] with Ratio & Sum Constraints, k={k} & x={x} & γ={gamma}")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('plot_fixed_w1.png', dpi=300)
    plt.show()
    print("Plot for fixed w[1] has been saved as 'plot_fixed_w1.png'.")

elif plot_mode == 'both':
    # Plot both cases side by side in one figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left subplot: fixed w[0] -> allowed region for w[1]
    ax1.fill_between(w0, eff_lower_w1, eff_upper_w1, color='lightblue', alpha=0.5,
                     label='Allowed region for w[1]')
    ax1.plot(w0, cand_lower_w1, 'b--', label=f'Lower: {gamma}^({k}+{x}) = {r_low:.3f} * w[0]')
    ax1.plot(w0, cand_upper_w1, 'r--', label=f'Upper: {gamma}^({k - 1}) = {r_high:.3f} * w[0]')
    ax1.plot(w0, 1 - w0, 'k-', label='w[0]+w[1]=1')
    ax1.set_xlabel('Fixed weight w[0]')
    ax1.set_ylabel('Computed weight w[1]')
    plt.title(f"Allowed w[1] vs. w[0], k={k} & x={x} & γ={gamma}")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.legend()
    ax1.grid(True)

    # Right subplot: fixed w[1] -> allowed region for w[0]
    ax2.fill_between(w1, eff_lower_w0, eff_upper_w0, color='lightgreen', alpha=0.5,
                     label='Allowed region for w[0]')
    ax2.plot(w1, cand_lower_w0, 'g--', label=f'Lower: w[1]/{gamma}^({k - 1})')
    ax2.plot(w1, cand_upper_w0, 'm--', label=f'Upper: w[1]/{gamma}^({k}+{x})')
    ax2.plot(w1, 1 - w1, 'k-', label='w[0]+w[1]=1')
    ax2.set_xlabel('Fixed weight w[1]')
    ax2.set_ylabel('Computed weight w[0]')
    plt.title(f"Allowed w[1] vs. w[0], k={k} & x={x} & γ={gamma}")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('combined_weight_ranges.png', dpi=300)
    plt.show()
    print("Combined plot has been saved as 'combined_weight_ranges.png'.")

else:
    print("Unknown plot_mode option. Please set plot_mode to 'w0', 'w1', or 'both'.")
