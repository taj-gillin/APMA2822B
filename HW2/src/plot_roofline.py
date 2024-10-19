import matplotlib.pyplot as plt
import numpy as np

# CPU specifications
peak_memory_bandwidth = 160  # GB/s
peak_flop_rate = 1000  # GFLOP/s (1 TFLOP/s)

# Measured bandwidths
main_loop_bandwidth = 55.3977  # GB/s
convergence_loop_bandwidth = 36.2605  # GB/s

# Arithmetic intensities
main_loop_ai = 0.167  # FLOPS/byte
convergence_loop_ai = 0.188  # FLOPS/byte

# Calculate achieved performance
main_loop_performance = main_loop_bandwidth * main_loop_ai
convergence_loop_performance = convergence_loop_bandwidth * convergence_loop_ai

# Create the roofline model
ai_range = np.logspace(-2, 2, 1000)
memory_bound = peak_memory_bandwidth * ai_range
compute_bound = np.full_like(ai_range, peak_flop_rate)
roofline = np.minimum(memory_bound, compute_bound)

# Plot the roofline model
plt.figure(figsize=(12, 8))
plt.loglog(ai_range, roofline, 'b-', linewidth=2, label='Roofline')
plt.loglog(ai_range, memory_bound, 'b--', linewidth=1, label='Memory Bound')
plt.loglog(ai_range, compute_bound, 'b--', linewidth=1, label='Compute Bound')

# Plot the measured performance points
plt.plot(main_loop_ai, main_loop_performance, 'ro', markersize=10, label='Main Loop')
plt.plot(convergence_loop_ai, convergence_loop_performance, 'go', markersize=10, label='Convergence Loop')

# Customize the plot
plt.xlabel('Arithmetic Intensity (FLOPS/byte)')
plt.ylabel('Performance (GFLOPS/s)')
plt.title('Roofline Model')
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.5)

# Set axis limits
plt.xlim(0.01, 100)
plt.ylim(1, 2000)

# Add annotations
plt.annotate(f'Main Loop: {main_loop_performance:.2f} GFLOPS/s', 
             xy=(main_loop_ai, main_loop_performance), xytext=(0.3, 20),
             arrowprops=dict(facecolor='red', shrink=0.05))
plt.annotate(f'Convergence Loop: {convergence_loop_performance:.2f} GFLOPS/s', 
             xy=(convergence_loop_ai, convergence_loop_performance), xytext=(0.3, 10),
             arrowprops=dict(facecolor='green', shrink=0.05))

# Save the plot
plt.savefig('roofline_model.png', dpi=300, bbox_inches='tight')
# plt.savefig('roofline_model.png', dpi=300, bbox_inches='tight')
