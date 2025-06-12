import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from allensdk.core.brain_observatory_cache import BrainObservatoryCache
from allensdk.brain_observatory.locally_sparse_noise import LocallySparseNoise

# Load Brain Observatory Cache
manifest_file = '../../../data/allen-brain-observatory/visual-coding-2p/manifest.json'
boc = BrainObservatoryCache(manifest_file=manifest_file)

# Load the cell specimen table for filtering
cell_specimen_table = pd.DataFrame(boc.get_cell_specimens())

# Filter for VISp and imaging depth between 300 and 550 um
exps = boc.get_ophys_experiments(
    targeted_structures=['VISp'],
    imaging_depths=[300, 350, 400, 450, 500, 550],
    stimuli=['locally_sparse_noise'],
)
if len(exps) == 0:
    raise RuntimeError("No experiments found")

# Choose the first matching experiment
exp = exps[0]
session_id = exp['id']
cre_line = exp['cre_line']
depth = exp['imaging_depth']
print(f"Session {session_id}, Cre-line: {cre_line}, Depth: {depth}µm (VISp)")

# Load the dataset
dset = boc.get_ophys_experiment_data(ophys_experiment_id=session_id)
ts, dff = dset.get_dff_traces()
cell_ids = dset.get_cell_specimen_ids()

lsn_table = dset.get_stimulus_table('locally_sparse_noise')
starts = lsn_table.start.values
ends = lsn_table.end.values
frames = lsn_table.frame.values


# Build receptive field map
templ = dset.get_stimulus_template('locally_sparse_noise')

H, W = templ.shape[1], templ.shape[2]
n_cells = dff.shape[0]
rf_maps = np.zeros((H, W, n_cells))
print(H)
print(W)

#stim_duration_sec = lsn_table["end"].iloc[0] - lsn_table["start"].iloc[0]

#print(lsn_table.head)
#frame_rate = lsn_table["frame"].iloc[0]
#window_duration = int(np.round(stim_duration_sec * frame_rate))
window_duration = lsn_table["end"].iloc[0] - lsn_table["start"].iloc[0]

print(f"Window duration (frames): {window_duration}")

print("max frame index:", np.max(frames))
print("min frame index:", np.min(frames))
print(f"H*W = {H*W}")

print(len(frames))

for trial_idx, frm in enumerate(frames):
    # stimulus at this frame: shape (H, W)
    stim_frame = templ[frm, :, :]  # 16 x 28

    s = starts[trial_idx]
    e = s + window_duration

    # average cell response in response window
    slice_resp = dff[:, s:e].mean(axis=1)  # n_cells

    # For all pixels in this frame that are ON
    on_pixels = np.argwhere(stim_frame != 0)

    for (y, x) in on_pixels:
        rf_maps[y, x, :] += slice_resp


# Normalize each RF map across space
rf_maps_mean = rf_maps.mean(axis=(0, 1))
rf_maps_std = rf_maps.std(axis=(0, 1))
rf_maps_std[rf_maps_std == 0] = 1e-6  # avoid division by zero

rf_maps -= rf_maps_mean
rf_maps /= rf_maps_std

# Plot example RFs
n_plot = min(16, n_cells)
fig, axes = plt.subplots(4, 4, figsize=(10, 10))
axes = axes.flatten()

# Store last imshow result for colorbar
for i in range(n_plot):
    im = axes[i].imshow(rf_maps[:, :, i], cmap='RdBu_r', origin='lower', vmin=-2, vmax=2)
    axes[i].set_title(f"Cell {cell_ids[i]}")
    axes[i].axis('off')

# Adjust space to make room for colorbar
fig.subplots_adjust(right=0.85)

# Create new axis for colorbar
cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
fig.colorbar(im, cax=cbar_ax)

plt.suptitle(f"Receptive Fields (VISp) - {cre_line} @ {depth}µm", y=1.02)
plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.show()
