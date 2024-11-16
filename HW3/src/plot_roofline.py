import matplotlib.pyplot as plt
import numpy as np

# GPU specifications from the output
peak_memory_bandwidth = 1638.4  # GB/s from output
peak_flop_rate = 22630.4  # GFLOPS from output

# Measured values from output
main_loop_bandwidth = 124.521  # GB/s
convergence_loop_bandwidth = 12.6253  # GB/s

# Arithmetic intensities from output
main_loop_ai = 0.25  # FLOPS/byte
convergence_loop_ai = 0.1875  # FLOPS/byte

# Calculate achieved performance (bandwidth * AI)
main_loop_performance = main_loop_bandwidth * main_loop_ai
convergence_loop_performance = convergence_loop_bandwidth * convergence_loop_ai

# Create the roofline model with better range and resolution
ai_range = np.logspace(-3, 4, 1000)  # Wider range for better visibility
memory_bound = peak_memory_bandwidth * ai_range
compute_bound = np.full_like(ai_range, peak_flop_rate)
roofline = np.minimum(memory_bound, compute_bound)

# Improve plot aesthetics
plt.figure(figsize=(12, 8))
plt.loglog(ai_range, roofline, 'b-', linewidth=2, label='Roofline')
plt.loglog(ai_range, memory_bound, 'b--', alpha=0.5, label='Memory Bound')
plt.loglog(ai_range, compute_bound, 'b--', alpha=0.5, label='Compute Bound')

# Add measured points with better visibility
plt.plot(main_loop_ai, main_loop_performance, 'ro', markersize=10, 
         label='Main Loop', zorder=5)
plt.plot(convergence_loop_ai, convergence_loop_performance, 'go', 
         markersize=10, label='Convergence Loop', zorder=5)

# Better annotations
plt.annotate(f'Main Loop\n{main_loop_performance:.1f} GFLOP/s\nAI: {main_loop_ai:.2f}', 
             xy=(main_loop_ai, main_loop_performance), 
             xytext=(main_loop_ai*2, main_loop_performance*2),
             arrowprops=dict(facecolor='red', shrink=0.05))

plt.annotate(f'Convergence Loop\n{convergence_loop_performance:.1f} GFLOP/s\nAI: {convergence_loop_ai:.2f}', 
             xy=(convergence_loop_ai, convergence_loop_performance),
             xytext=(convergence_loop_ai*0.5, convergence_loop_performance*2),
             arrowprops=dict(facecolor='green', shrink=0.05))

# Better axis labels and title
plt.xlabel('Arithmetic Intensity (FLOPS/byte)', fontsize=12)
plt.ylabel('Performance (GFLOP/s)', fontsize=12)
plt.title('GPU Roofline Model Analysis', fontsize=14)
plt.grid(True, which="both", ls="-", alpha=0.2)

# Adjust limits for better visibility
plt.xlim(1e-1, 1e2)
plt.ylim(1, 5e4)

# Calculate and plot ridge point
# ridge_point = peak_flop_rate / peak_memory_bandwidth
# plt.plot(ridge_point, peak_flop_rate, 'k*', markersize=15, label='Ridge Point')
# plt.annotate(f'Ridge Point\nAI: {ridge_point:.1f}', 
#              xy=(ridge_point, peak_flop_rate),
#              xytext=(ridge_point*2, peak_flop_rate*0.8),
#              arrowprops=dict(facecolor='black', shrink=0.05))

# Save the plot
plt.savefig('gpu_roofline_model.png', dpi=300, bbox_inches='tight')

# Print analysis
print(f"Performance Analysis:")
print(f"Main Loop:")
print(f"  - Arithmetic Intensity: {main_loop_ai:.2f} FLOPS/byte")
print(f"  - Performance: {main_loop_performance:.2f} GFLOP/s")
print(f"  - Efficiency: {(main_loop_performance/peak_flop_rate)*100:.1f}% of peak")
print(f"\nConvergence Loop:")
print(f"  - Arithmetic Intensity: {convergence_loop_ai:.2f} FLOPS/byte")
print(f"  - Performance: {convergence_loop_performance:.2f} GFLOP/s")
print(f"  - Efficiency: {(convergence_loop_performance/peak_flop_rate)*100:.1f}% of peak")
