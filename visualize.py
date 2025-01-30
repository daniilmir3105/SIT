import os
import numpy as np
import matplotlib.pyplot as plt

def visualize_seismic(filepath):
    """
Visualize a seismic data file as a heatmap.

Parameters:
filepath (str): Path to the .npy file containing seismic data.
    """
    # Load seismic data
    seismic_data = np.load(filepath)

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.imshow(seismic_data, cmap='gray', aspect='auto', vmin=-1, vmax=1)
    plt.title("Исходные данные")
    plt.xlabel("Расстояние от источника, м")
    plt.ylabel("Время свободного пробега, мс")
    plt.colorbar(label='Амплитуда')
    plt.show()

# Path to the folder containing seismic data
# data_folder = "test_data/subsampled"
current_dir = os.path.dirname(os.path.abspath(__file__))
data_folder = os.path.join(current_dir, "data")
# data_folder = os.path.join(current_dir, "subsampled")

# Choose one file to visualize
file_to_visualize = None
if os.path.exists(data_folder):
    files = [f for f in os.listdir(data_folder) if f.endswith(".npy")]
    if files:
        file_to_visualize = os.path.join(data_folder, files[0])  # Visualize the first file

if __name__ == "__main__":
    # visualize_seismic(file_to_visualize)
    # Visualize the file
    if file_to_visualize:
        visualize_seismic(file_to_visualize)
    else:
        print(f"No .npy files found in {data_folder}.")