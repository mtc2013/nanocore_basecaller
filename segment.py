import numpy as np
import matplotlib.pyplot as plt

def segment_signal(signal, window_size=10, threshold=15.0):
    """
    Segments a continuous nanopore signal into discrete events 
    using a rolling t-test/change-point detection method.
    """
    events = []
    start_idx = 0
    n = len(signal)
    
    # Iterate through the signal leaving room for leading and trailing windows
    i = window_size
    while i < n - window_size:
        # Window 1: Data right before the split point
        w1 = signal[i - window_size : i]
        # Window 2: Data right after the split point
        w2 = signal[i : i + window_size]
        
        # Calculate means and variances of both windows
        m1, m2 = np.mean(w1), np.mean(w2)
        v1, v2 = np.var(w1) + 1e-5, np.var(w2) + 1e-5 # add small epsilon to avoid div by zero
        
        # Student's t-statistic formulation for difference between two means
        t_stat = np.abs(m1 - m2) / np.sqrt((v1 / window_size) + (v2 / window_size))
        
        # If t-stat exceeds threshold, we found a sudden shift in current!
        if t_stat > threshold:
            # Extract the raw block belonging to this segment
            segment_data = signal[start_idx:i]
            
            # Collapse the segment into an Event (continuous Mean + variable Duration)
            events.append({
                "mean": np.mean(segment_data),
                "duration": i - start_idx,
                "start": start_idx,
                "end": i
            })
            
            # Reset the next segment's start point and skip ahead past the window
            start_idx = i
            i += window_size
        else:
            i += 1
            
    # Catch the remaining tail of the signal
    if start_idx < n:
        events.append({
            "mean": np.mean(signal[start_idx:]),
            "duration": n - start_idx,
            "start": start_idx,
            "end": n
        })
        
    return events


import os
import hdf5plugin

import matplotlib.pyplot as plt

import vbz_h5py_plugin

plugin_dir = os.path.join(os.path.dirname(vbz_h5py_plugin.__file__))
os.environ["HDF5_PLUGIN_PATH"] = plugin_dir

import h5py
import numpy as np

file_path = "PAG65784_pass_f306681d_16a70748_0.fast5"

with h5py.File(file_path, 'r') as f:
    read_ids = list(f.keys())
    first_read = read_ids[0]
    
    signal_ds = f[first_read]["Raw"]["Signal"]
    
    # Inspect the dataset before reading
    print(f"Shape: {signal_ds.shape}")
    print(f"Dtype: {signal_ds.dtype}")
    print(f"Chunks: {signal_ds.chunks}")
    
    # This should now change from "unknown" to a tracking ID integer
    print(f"Compression ID: {signal_ds.compression}")
    
    # Read the data natively using h5py's slicing syntax (cleaner than np.array)
    raw_signal = signal_ds[...] 

# Run segmentation
discovered_events = segment_signal(raw_signal, window_size=5, threshold=4.5)

# Print the collapsed sequence
print(f"Original Signal Length: {len(raw_signal)} points")
print(f"Collapsed Event Sequence Length: {len(discovered_events)} Events\n")
for idx, ev in enumerate(discovered_events):
    print(f"Event {idx+1}: Mean Current = {ev['mean']:.2f} | Duration = {ev['duration']} points")

# --- VISUALIZE WHAT SEGMENTATION ACTUALLY LOOKS LIKE ---
plt.figure(figsize=(10, 4))
plt.plot(raw_signal, color='lightgray', label='Raw Continuous Points')

# Draw the segmented steps
for ev in discovered_events:
    plt.hlines(ev['mean'], ev['start'], ev['end'], colors='red', linewidth=3)
    plt.axvline(ev['end'], color='blue', linestyle='--', alpha=0.3)

plt.title("Nanopore Event Segmentation (Time-Axis Compression)")
plt.ylabel("Ionic Current (Y)")
plt.xlabel("Raw Sequential Sample Steps (X)")
plt.legend(["Raw Signal", "Segmented Step Function (Continuous Means)"])
plt.xlim(2000, 2300)
plt.show()