from DataReader import DataReader
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import time

# Density-based approach - faster for very high dimensional cases
def analyze_hypercube_density(data, num_bins=5, max_samples=1000000):
    """
    Analyzes the density and distribution of data points in hypercube space.
    This is useful when D^m is extremely large.
    
    Parameters:
    - data: numpy array of shape (N, D)
    - num_bins: number of bins per dimension
    - max_samples: maximum number of points to process (for performance)
    
    Returns:
    - density statistics
    """
    N, D = data.shape
    print(f"Analyzing hypercube density with {min(N, max_samples):,} points")
    
    # Subsample if necessary
    if N > max_samples:
        indices = np.random.choice(N, max_samples, replace=False)
        data_sample = data[indices]
    else:
        data_sample = data
    
    # Find min and max for each dimension
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    range_vals = max_vals - min_vals
    epsilon = 1e-10
    
    # Normalize and bin the data
    normalized_data = (data_sample - min_vals) / (range_vals + epsilon)
    bin_indices = np.floor(normalized_data * num_bins).astype(int)
    bin_indices = np.clip(bin_indices, 0, num_bins - 1)
    
    # Count points per hypercube
    hypercube_counts = Counter(tuple(bin_indices[i]) for i in range(len(data_sample)))
    
    # Calculate statistics
    num_occupied = len(hypercube_counts)
    max_density = max(hypercube_counts.values())
    avg_density = len(data_sample) / num_occupied if num_occupied > 0 else 0
    
    total_hypercubes = num_bins ** D
    occupancy_rate = num_occupied / total_hypercubes * 100
    
    print(f"Number of occupied hypercubes: {num_occupied:,} out of {total_hypercubes:,}")
    print(f"Occupancy rate: {occupancy_rate:.8f}%")
    print(f"Average points per occupied hypercube: {avg_density:.2f}")
    print(f"Maximum points in any hypercube: {max_density}")
    
    # Estimate empty hypercubes
    empty_hypercubes = total_hypercubes - num_occupied
    empty_percentage = 100 - occupancy_rate
    
    print(f"Estimated empty hypercubes: {empty_hypercubes:,}")
    print(f"Estimated empty percentage: {empty_percentage:.10f}%")
    
    return {
        "occupied_hypercubes": num_occupied,
        "empty_hypercubes": empty_hypercubes,
        "empty_percentage": empty_percentage,
        "avg_density": avg_density,
        "max_density": max_density,
        "hypercubes": hypercube_counts,
        "hypercube_indices": bin_indices
    }

def find_similar_profiles(x_profile,model,log=True):
    # Get the fixed features (all except M_contribution and V_contribution)
    fixed_features = [feat for feat in model.features if feat not in ['M_contribution', 'V_contribution']]
    
    # Create a hashable key from the fixed features of x_profile
    x_profile_key = tuple(x_profile[fixed_features].values.astype(np.float32))[0]
    
    # Extract all profile data as a DataFrame for faster processing
    all_profiles_df = pd.DataFrame(model.x_data, columns=model.features)
    
    # Create mask for matching profiles
    mask = np.ones(len(all_profiles_df), dtype=bool)
    
    # Apply filter for each fixed feature
    for i, feature in enumerate(fixed_features):
        feature_idx = model.features.index(feature)
        mask &= (all_profiles_df.iloc[:, feature_idx] == x_profile_key[i])
    
    # Get indices of matching profiles
    profiles_similar = np.where(mask)[0].tolist()
    
    if log==True:
        print(f'{len(profiles_similar)} profiles found with the same fixed features')
    
    return profiles_similar
