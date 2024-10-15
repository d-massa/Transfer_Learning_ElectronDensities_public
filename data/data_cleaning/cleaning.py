import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.fft import fft2, fftshift
import re
from joblib import Parallel, delayed

DB_3D = pd.read_csv('FCC_diff_3D.csv')
DB_3D.set_index(['Unnamed: 0'], inplace=True)

def extract_sample_ids_from_filenames(folder):
    """Extract sample IDs from filenames in the specified folder."""
    pattern = re.compile(r'normalized_data_(mp-\d+)\.png$')  # Regex to match mp-# in filenames
    sample_ids = set()
    
    for filename in os.listdir(folder):
        match = pattern.search(filename)
        if match:
            sample_ids.add(match.group(1))
    
    print("Extracted Sample IDs:", sample_ids)  # Debug: Print extracted sample IDs
    return sample_ids

def filter_csv_files(sample_ids, description_file, properties_file, output_folder):
    """Filter FCC descriptions and properties CSV files based on sample IDs."""
    # Load the CSV files
    descriptions_df = pd.read_csv(description_file, index_col=0)
    properties_df = pd.read_csv(properties_file, index_col=0)
    
    # Ensure index is of string type
    descriptions_df.index = descriptions_df.index.astype(str)
    properties_df.index = properties_df.index.astype(str)
    
    # Debug: Print the first few indexes to check for format issues
    print("Description CSV Indexes:", descriptions_df.index[:5])
    print("Properties CSV Indexes:", properties_df.index[:5])

    # Filter based on sample IDs
    filtered_descriptions = descriptions_df.loc[descriptions_df.index.intersection(sample_ids)]
    filtered_properties = properties_df.loc[properties_df.index.intersection(sample_ids)]
    
    # Debug: Check if filtering is working correctly
    print("Filtered Descriptions Shape:", filtered_descriptions.shape)
    print("Filtered Properties Shape:", filtered_properties.shape)
    
    # Save the filtered data to new CSV files
    filtered_descriptions.to_csv(os.path.join(output_folder, 'filtered_FCC_descriptions.csv'))
    filtered_properties.to_csv(os.path.join(output_folder, 'filtered_FCC_properties.csv'))

# Define normalization methods (Vectorized where possible)
def min_max_normalize(data, min_val, max_val):
    return (data - min_val) / (max_val - min_val)

def z_score_normalize(data, mean_val, std_dev):
    return (data - mean_val) / std_dev

def robust_scale(data, median_val, iqr):
    return (data - median_val) / iqr

def unit_vector_normalize(data, norm):
    return data / norm

def maxabs_scale(data, max_abs):
    return data / max_abs

def zscore_interval(data, mean_val=None, std_dev=None, interval=(-1, 1)):
    mean_val = mean_val or np.mean(data)
    std_dev = std_dev or np.std(data)
    z_score_normalized = (data - mean_val) / std_dev
    min_interval, max_interval = interval
    min_z, max_z = np.min(z_score_normalized), np.max(z_score_normalized)
    scaled_data = (z_score_normalized - min_z) / (max_z - min_z)
    return scaled_data * (max_interval - min_interval) + min_interval  

# Plotting functions
def plot_image(data, folder, filename, cmap='viridis'):
    plt.imshow(data, cmap=cmap, origin='lower')
    plt.colorbar(label='Normalized Electron Charge Density')
    plt.savefig(os.path.join(folder, filename))
    plt.close()

def plot_image_noborders(data, folder, filename, cmap='viridis'):
    # Save the image directly with plt.imsave
    plt.imsave(os.path.join(folder, filename), data, cmap=cmap, origin='lower')

def plot_fourier_transform(data, folder, filename):
    f_transform = fft2(data)
    magnitude_spectrum = np.abs(fftshift(f_transform))
    plt.imshow(np.log1p(magnitude_spectrum), cmap='gray')
    plt.colorbar(label='Log Magnitude Spectrum')
    plt.savefig(os.path.join(folder, filename))
    plt.close()

def plot_fourier_transform_noborders(data, folder, filename):
    f_transform = fft2(data)
    magnitude_spectrum = np.abs(fftshift(f_transform))
    log_magnitude = np.log1p(magnitude_spectrum)
    plt.imsave(os.path.join(folder, filename), log_magnitude, cmap='gray', origin='lower')

# Data cleaning function
def clean_data(data_samples, max_vals, min_vals, boundary_factor=1.5):
    def calculate_iqr_boundaries(values, factor):
        Q1, Q3 = np.percentile(values, [25, 75])
        IQR = Q3 - Q1
        return Q1 - factor * IQR, Q3 + factor * IQR

    max_lower_bound, max_upper_bound = calculate_iqr_boundaries(max_vals, boundary_factor)
    min_lower_bound, min_upper_bound = calculate_iqr_boundaries(min_vals, boundary_factor)

    cleaned_samples = [
        (sample_id, grid_z_square) for sample_id, grid_z_square in data_samples
        if max_lower_bound <= np.max(grid_z_square) <= max_upper_bound and
           min_lower_bound <= np.min(grid_z_square) <= min_upper_bound
    ]

    # Plot maxima and minima distributions
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.hist(max_vals, bins=50, color='blue', alpha=0.7)
    plt.axvline(max_lower_bound, color='red', linestyle='--')
    plt.axvline(max_upper_bound, color='red', linestyle='--')
    plt.title('Maxima Distribution with Boundaries')
    plt.xlabel('Max Value')
    plt.ylabel('Frequency')

    plt.subplot(1, 2, 2)
    plt.hist(min_vals, bins=50, color='green', alpha=0.7)
    plt.axvline(min_lower_bound, color='red', linestyle='--')
    plt.axvline(min_upper_bound, color='red', linestyle='--')
    plt.title('Minima Distribution with Boundaries')
    plt.xlabel('Min Value')
    plt.ylabel('Frequency')

    plt.suptitle('Distribution of Maxima and Minima with IQR Boundaries')
    plt.savefig(os.path.join(base_folder, 'maxima_minima_distribution.png'))
    plt.close()

    return cleaned_samples

# Process individual samples, including data cleaning
def process_sample(index, sample_ids, DB_3D, x_min, x_max, y_min, y_max):
    sample_id = sample_ids[index]
    data = DB_3D.to_numpy()[index].reshape(60, 60, 60)
    extended_data = np.tile(data, (8, 8, 8))

    new_P = 480
    x, y, z = np.indices(extended_data.shape)
    mask = (x + y + z) == new_P
    x_coords_ext, y_coords_ext, values_ext = x[mask], y[mask], extended_data[mask]

    grid_x, grid_y = np.mgrid[x_coords_ext.min():x_coords_ext.max():480j, y_coords_ext.min():y_coords_ext.max():480j]
    grid_z = griddata((x_coords_ext, y_coords_ext), values_ext, (grid_x, grid_y), method='cubic')

    x_indices = np.where((grid_x[:, 0] >= x_min) & (grid_x[:, 0] <= x_max))[0]
    y_indices = np.where((grid_y[0, :] >= y_min) & (grid_y[0, :] <= y_max))[0]
    grid_z_square = grid_z[x_indices.min():x_indices.max() + 1, y_indices.min():y_indices.max() + 1]

    max_val, min_val = np.max(grid_z_square), np.min(grid_z_square)
    return (sample_id, grid_z_square), max_val, min_val

# Normalization and saving functions
def normalize_and_save(sample_id, grid_z_square, method, normalization_params, folder_structure):
    min_val, max_val, mean_val, std_dev, median_val, iqr, norm, max_abs = normalization_params

    if method == 'minmax':
        normalized_data = min_max_normalize(grid_z_square, min_val, max_val)
    elif method == 'zscore':
        normalized_data = z_score_normalize(grid_z_square, mean_val, std_dev)
    elif method == 'robust':
        normalized_data = robust_scale(grid_z_square, median_val, iqr)
    elif method == 'unit_vector':
        normalized_data = unit_vector_normalize(grid_z_square, norm)
    elif method == 'maxabs':
        normalized_data = maxabs_scale(grid_z_square, max_abs)
    elif method == 'zscore_interval':
        normalized_data = zscore_interval(grid_z_square, mean_val, std_dev, interval=(-1, 1))
    else:
        raise ValueError(f"Normalization method '{method}' not recognized.")

    plot_image_noborders(normalized_data, folder_structure['base'], f'normalized_data_{sample_id}.png')
    plot_image_noborders(normalized_data, folder_structure['grey'], f'grayscale_data_{sample_id}.png', cmap='gray')
    plot_fourier_transform_noborders(normalized_data, folder_structure['ft'], f'fourier_transform_{sample_id}.png')


method = 'zscore_interval'
base_folder = f'Newdata_normalized_{method}'
os.makedirs(base_folder, exist_ok=True)
folder_structure = {
    'base': base_folder,
    'grey': os.path.join(base_folder, 'GREY'),
    'ft': os.path.join(base_folder, 'FT')
}
for folder in folder_structure.values():
    os.makedirs(folder, exist_ok=True)

sample_ids = DB_3D.index.tolist()
x_min, x_max, y_min, y_max = 25, 210, 25, 210

# First Pass: Extract Data and Calculate Max/Min Values
results = Parallel(n_jobs=32, backend="multiprocessing")(
    delayed(process_sample)(
        index, sample_ids, DB_3D, x_min, x_max, y_min, y_max
    ) for index in range(len(sample_ids))
)

data_samples = [res[0] for res in results]
max_vals = [res[1] for res in results]
min_vals = [res[2] for res in results]

# Data Cleaning
cleaned_data_samples = clean_data(data_samples, max_vals, min_vals, boundary_factor=1.5)

# Prepare normalization parameters
dataset = np.concatenate([grid_z_square.flatten() for _, grid_z_square in cleaned_data_samples])
normalization_params = (
    np.min(dataset), np.max(dataset),
    np.mean(dataset), np.std(dataset),
    np.median(dataset), np.percentile(dataset, 75) - np.percentile(dataset, 25),
    np.linalg.norm(dataset), np.max(np.abs(dataset))
)

# Second Pass: Normalization and Saving
Parallel(n_jobs=8, backend="multiprocessing")(
    delayed(normalize_and_save)(
        sample_id, grid_z_square, method, normalization_params, folder_structure
    ) for sample_id, grid_z_square in cleaned_data_samples
)

print("Processing complete.")



# Define paths
image_folder = base_folder  # The folder with the saved images
description_csv = 'FCC_descriptions.csv'
properties_csv = 'FCC_properties_w_dens.csv'
output_folder = base_folder  # or specify a different folder for filtered files

# Extract sample IDs
sample_ids = extract_sample_ids_from_filenames(image_folder)

# Filter and save CSV files
filter_csv_files(sample_ids, description_csv, properties_csv, output_folder)

print("Filtering complete.")

