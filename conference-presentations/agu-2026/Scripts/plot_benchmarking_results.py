import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

fname = '/Users/jschaper/Documents/Codes/gesdisc-tutorials/conference-presentations/agu-2026/Data/access_benchmark_results_nw_op8_nw_co9.csv'

# Read the CSV file
# Replace 'your_data.csv' with your actual CSV filename
df = pd.read_csv(fname, sep='\t', skiprows=1, 
                 names=['Method', 'Days', 'Trial', 'Lazy_Load_Time_s', 'Compute_Time_s', 
                        'Estimated_Data_Size_MiB', 'HTTP_Retries', 'Token_Refreshes', 
                        'NetCDF_File', 'Status'])

print("Data loaded:")
print(df.head())
print("\nData types:")
print(df.dtypes)

# Convert numeric columns to proper types
df['Days'] = pd.to_numeric(df['Days'])
df['Lazy_Load_Time_s'] = pd.to_numeric(df['Lazy_Load_Time_s'])
df['Compute_Time_s'] = pd.to_numeric(df['Compute_Time_s'])

# Calculate total time
df['Total_Time_s'] = df['Lazy_Load_Time_s'] + df['Compute_Time_s']

# Group by Method and Days to get mean and std
summary = df.groupby(['Method', 'Days']).agg({
    'Lazy_Load_Time_s': ['mean', 'std'],
    'Compute_Time_s': ['mean', 'std'],
    'Total_Time_s': ['mean', 'std']
}).reset_index()

print("\nSummary before flattening:")
print(summary.head())

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Kerchunk vs OPeNDAP Performance Comparison', fontsize=16, fontweight='bold')

# Get unique days and methods
days = sorted(df['Days'].unique())
methods = df['Method'].unique()
colors = {'Kerchunk': '#1f77b4', 'OPeNDAP': '#ff7f0e'}

# Plot 1: Total Time Comparison
ax1 = axes[0, 0]
for method in methods:
    method_data = summary[summary['Method'] == method]
    ax1.errorbar(method_data['Days'], method_data[('Total_Time_s', 'mean')], 
                 yerr=method_data[('Total_Time_s', 'std')], 
                 marker='o', label=method, capsize=5, linewidth=2, markersize=8,
                 color=colors.get(method, 'gray'))
ax1.set_xlabel('Days', fontsize=12)
ax1.set_ylabel('Total Time (s)', fontsize=12)
ax1.set_title('Total Time (Lazy Load + Compute)', fontsize=13, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Lazy Load Time Comparison
ax2 = axes[0, 1]
for method in methods:
    method_data = summary[summary['Method'] == method]
    ax2.errorbar(method_data['Days'], method_data[('Lazy_Load_Time_s', 'mean')], 
                 yerr=method_data[('Lazy_Load_Time_s', 'std')], 
                 marker='s', label=method, capsize=5, linewidth=2, markersize=8,
                 color=colors.get(method, 'gray'))
ax2.set_xlabel('Days', fontsize=12)
ax2.set_ylabel('Lazy Load Time (s)', fontsize=12)
ax2.set_title('Lazy Load Time', fontsize=13, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Compute Time Comparison
ax3 = axes[1, 0]
for method in methods:
    method_data = summary[summary['Method'] == method]
    ax3.errorbar(method_data['Days'], method_data[('Compute_Time_s', 'mean')], 
                 yerr=method_data[('Compute_Time_s', 'std')], 
                 marker='^', label=method, capsize=5, linewidth=2, markersize=8,
                 color=colors.get(method, 'gray'))
ax3.set_xlabel('Days', fontsize=12)
ax3.set_ylabel('Compute Time (s)', fontsize=12)
ax3.set_title('Compute Time', fontsize=13, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Stacked Bar Chart
ax4 = axes[1, 1]
x = np.arange(len(days))
width = 0.35

kerchunk_data = summary[summary['Method'] == 'Kerchunk'].sort_values('Days')
opendap_data = summary[summary['Method'] == 'OPeNDAP'].sort_values('Days')

# Stacked bars for Kerchunk
ax4.bar(x - width/2, kerchunk_data[('Lazy_Load_Time_s', 'mean')].values, width, 
        label='Kerchunk - Lazy Load', color='#1f77b4')
ax4.bar(x - width/2, kerchunk_data[('Compute_Time_s', 'mean')].values, width, 
        bottom=kerchunk_data[('Lazy_Load_Time_s', 'mean')].values,
        label='Kerchunk - Compute', color='#aec7e8')

# Stacked bars for OPeNDAP
ax4.bar(x + width/2, opendap_data[('Lazy_Load_Time_s', 'mean')].values, width, 
        label='OPeNDAP - Lazy Load', color='#ff7f0e')
ax4.bar(x + width/2, opendap_data[('Compute_Time_s', 'mean')].values, width, 
        bottom=opendap_data[('Lazy_Load_Time_s', 'mean')].values,
        label='OPeNDAP - Compute', color='#ffbb78')

ax4.set_xlabel('Days', fontsize=12)
ax4.set_ylabel('Time (s)', fontsize=12)
ax4.set_title('Time Breakdown by Component', fontsize=13, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(days)
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nSummary Statistics:")
print("="*80)
print(summary)