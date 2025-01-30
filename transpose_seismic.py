import os
import numpy as np

def transpose_seismic_data(data_folder):
    """
Transpose all seismic .npy files in the specified folder.

Parameters:
data_folder (str): Path to the folder containing .npy files.
    """
    # Check if the folder exists
    if not os.path.exists(data_folder):
        print(f"Folder {data_folder} does not exist.")
        return

    # Process each .npy file in the folder
    for filename in os.listdir(data_folder):
        if filename.endswith(".npy"):
            file_path = os.path.join(data_folder, filename)

            # Load the seismic data
            seismic_data = np.load(file_path)

            # Transpose the seismic data
            transposed_data = seismic_data.T

            # Save the transposed data back to the same file
            np.save(file_path, transposed_data)
            print(f"Transposed and saved: {filename}")

# Define the path to the data folder
current_dir = os.path.dirname(os.path.abspath(__file__))
data_folder = os.path.join(current_dir, "test_data")

if __name__ == "__main__":
    # Transpose all seismic data in the folder
    transpose_seismic_data(data_folder)