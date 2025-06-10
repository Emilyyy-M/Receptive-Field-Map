import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from allensdk.core.brain_observatory_cache import BrainObservatoryCache
from allensdk.brain_observatory.locally_sparse_noise import LocallySparseNoise

# Load Brain Observatory Cache
manifest_file = '../../../data/allen-brain-observatory/visual-coding-2p/manifest.json'
boc = BrainObservatoryCache(manifest_file=manifest_file)

# Step 1: Query all experiments matching VISp, LSN stimulus, imaging depth 300–550 um
experiments = boc.get_ophys_experiments(
    targeted_structures=['VISp'],
    stimuli=['locally_sparse_noise'],
)

filtered_experiments = [exp for exp in experiments if 300 <= exp['imaging_depth'] <= 550]

# Selecting the first matching experiment for now
experiment = filtered_experiments[0]
print(f"Using experiment ID: {experiment['id']}")
print(f"Cre line: {experiment['cre_line']}")
print(f"Imaging depth: {experiment['imaging_depth']} µm")

# Step 2: Load experiment data
data_set = boc.get_ophys_experiment_data(experiment['id'])

# Step 3: Use LocallySparseNoise analysis class
lsn = LocallySparseNoise(data_set)

# Get receptive field maps for all cells
receptive_fields = lsn.receptive_field

# Step 4: Visualize the receptive field map of a single cell
num_cells = receptive_fields.shape[0]
print(f"Number of cells: {num_cells}")

# Plot the first 10 valid receptive fields
num_to_plot = 10
plotted = 0
for cell_index in range(num_cells):
    if plotted >= num_to_plot:
        break

    rf = receptive_fields[cell_index]  # Shape: (2, H, W)

    # Skip empty or invalid RFs
    if rf is None or np.all(np.isnan(rf)):
        print(f"Skipping cell {cell_index}: RF is empty or invalid.")
        continue

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    # ON field
    axs[0].imshow(rf[0], cmap='hot', interpolation='nearest')
    axs[0].set_title(f"Cell {cell_index} - ON Receptive Field")
    axs[0].axis('off')

    # OFF field
    axs[1].imshow(rf[1], cmap='hot', interpolation='nearest')
    axs[1].set_title(f"Cell {cell_index} - OFF Receptive Field")
    axs[1].axis('off')

    plt.suptitle(f"Receptive Fields - Cell {cell_index}", fontsize=14)
    plt.tight_layout()
    plt.show()
    plotted += 1