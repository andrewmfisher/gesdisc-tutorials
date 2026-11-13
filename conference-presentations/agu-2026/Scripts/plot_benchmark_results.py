import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def create_box_plot(csv_filename, output_filename='benchmark_boxplot.png'):
    # 1. Load the data
    print(f"Loading data from {csv_filename}...")
    df = pd.read_csv(csv_filename)
    
    # Define the columns we want to plot
    # Change 'Compute_Time_s' to 'Lazy_Load_Time_s' if you prefer to plot that metric
    x_column = 'Method'
    y_column = 'Compute_Time_s'
    
    # 2. Clean the data
    # Ensure the target column is numeric (coercing errors to NaN)
    df[y_column] = pd.to_numeric(df[y_column], errors='coerce')
    
    # Drop rows where our target plotting data is missing (e.g., when compute was skipped)
    plot_data = df.dropna(subset=[x_column, y_column])
    
    # 3. Create the box and whisker plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=plot_data, x=x_column, y=y_column, palette="Set2")
    
    # Add optional swarmplot to show the individual data points on top of the boxes
    sns.stripplot(data=plot_data, x=x_column, y=y_column, color='black', alpha=0.5, size=4)
    
    # 4. Format the plot
    plt.title(f'Distribution of {y_column} by {x_column}', fontsize=14, pad=15)
    plt.xlabel('Access Method', fontsize=12)
    plt.ylabel('Time (seconds)', fontsize=12)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # 5. Save the plot
    print(f"Saving figure to {output_filename}...")
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    
    # 6. Display the plot
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Ensure this runs in your environment if that's where your packages are installed
    csv_file = '../Data/access_benchmark_results_00.csv'
    image_file = '../Figures/benchmark_boxplot.png'
    
    create_box_plot(csv_file, output_filename=image_file)
