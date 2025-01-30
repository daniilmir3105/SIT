import numpy as np


def preprocess_seismic_data(seismic_data, nsub, noise_level=0.1):
    """
    Preprocess seismic data by adding noise and subsampling traces.

    Parameters:
        seismic_data (np.ndarray): Original seismic data array.
        nsub (int): Subsampling factor. From every 'nsub' traces, only one is kept.
        noise_level (float): Standard deviation of the Gaussian noise to be added.

    Returns:
        np.ndarray: Preprocessed seismic data.
    """
    # Add Gaussian noise
    noise = np.random.normal(0, noise_level * np.std(seismic_data), seismic_data.shape)
    noisy_data = seismic_data + noise

    # Subsample traces by zeroing out
    # Determine which traces to keep
    kept_traces = np.arange(0, noisy_data.shape[1], nsub)
    # Zero out traces that are not in kept_traces
    subsampled_data = noisy_data.copy()
    for trace_idx in range(subsampled_data.shape[1]):
        if trace_idx not in kept_traces:
            subsampled_data[:, trace_idx] = 0

    return subsampled_data


# Load clear seismic data
x = np.load('test_data/clear.npy')
x = x.astype(np.float64)


# Define preprocessing parameters
nsub = 4  # From every 4 traces, keep 1
noise_level = 0  # 10% of data's standard deviation