import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Define machine characteristics
peak_bandwidth = 160  # GB/s
peak_performance = 1000  # GFLOP/s

# List of CSV files and their corresponding titles
csv_files = {
    'results/results_contiguous.csv': 'Roofline Model - Contiguous Memory Allocation',
    'results/results_separate_rows.csv': 'Roofline Model - Separate Row Allocations',
    'results/results_loop_unrolling.csv': 'Roofline Model - Loop Unrolling',
    'results/results_padding.csv': 'Roofline Model - Padding',
    'results/results_column_major.csv': 'Roofline Model - Column-Major Storage'
}

# Create a directory to save the plots
output_dir = 'roofline_plots'
os.makedirs(output_dir, exist_ok=True)

# Function to plot and save Roofline for a given CSV file
def plot_and_save_roofline(csv_file, title, output_path):
    # Load CSV data
    data = pd.read_csv(csv_file)
    
    # Ensure required columns are present
    required_columns = ['Arithmetic_Intensity(FLOP/Byte)', 'FLOP_Rate(GFLOPs/s)']
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found in {csv_file}. Please ensure the C++ script outputs this column.")
    
    # Extract AI and Performance
    ai = data['Arithmetic_Intensity(FLOP/Byte)']
    performance = data['FLOP_Rate(GFLOPs/s)']
    
    # Define AI range for Rooflines
    ai_mem = np.linspace(0, peak_performance / peak_bandwidth * 1.2, 500)  # Extend range slightly for better visualization
    perf_mem = peak_bandwidth * ai_mem  # GFLOP/s
    
    # Compute the intersection point
    ai_intersect = peak_performance / peak_bandwidth
    perf_intersect = peak_performance
    
    # Plotting
    plt.figure(figsize=(10, 7))
    
    # Plot data points
    plt.scatter(ai, performance, label='Data Points', color='blue')
    
    # Plot memory bandwidth roof
    plt.plot(ai_mem, perf_mem, label=f'Memory Bandwidth Roof ({peak_bandwidth} GB/s)', color='red')
    
    # Plot compute roof
    plt.axhline(y=peak_performance, color='green', linestyle='--', label=f'Compute Roof ({peak_performance/1000} TFLOP/s)')
    
    # Fill between memory and compute roofs
    plt.fill_between(ai_mem, perf_mem, peak_performance, where=(ai_mem <= ai_intersect), color='gray', alpha=0.2)
    
    # Labels and title
    plt.xlabel('Arithmetic Intensity (FLOP/Byte)', fontsize=14)
    plt.ylabel('Performance (GFLOP/s)', fontsize=14)
    plt.title(title, fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Set limits
    plt.xlim(0, max(ai_mem))
    plt.ylim(0, peak_performance * 1.1)
    
    # Save the plot as a PNG file
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()  # Close the figure to free memory
    
    print(f"Saved Roofline model to {output_path}")

# Iterate through each CSV file and generate the corresponding Roofline plot
for csv, title in csv_files.items():
    if not os.path.isfile(csv):
        print(f"Warning: {csv} not found. Skipping...")
        continue
    output_filename = os.path.splitext(csv)[0] + '.png'
    output_path = os.path.join(output_dir, output_filename)
    plot_and_save_roofline(csv, title, output_path)

print(f"All Roofline plots have been saved in the '{output_dir}' directory.")
