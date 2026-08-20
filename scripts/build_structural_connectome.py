"""Build the cached structural connectome used by notebooks 7 and 8.

Downloads DIPY's Stanford HARDI single-subject diffusion dataset (small,
free, no login), fits a tensor model and a constrained spherical
deconvolution model, runs deterministic whole-brain tractography, and
counts streamlines between region pairs of a Desikan-Killiany-style
parcellation that ships already registered to the same diffusion volume.

This is the heaviest computation in the whole series (whole-brain
tractography), which is exactly why it lives in a script instead of a
notebook: run it once here, and every notebook that needs the result
loads the small cached files this script writes to data/.

Run once:  python scripts/build_structural_connectome.py
Takes about one to two minutes on a laptop CPU.
"""

import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from dipy.data import read_stanford_labels, default_sphere
from dipy.direction import peaks_from_model
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel, auto_response_ssst
from dipy.reconst.dti import TensorModel, fractional_anisotropy
from dipy.tracking import utils
from dipy.tracking.local_tracking import LocalTracking
from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion
from dipy.tracking.streamline import Streamlines

OUT = "data"
os.makedirs(OUT, exist_ok=True)

# label id -> (name, new_label_id) from DIPY's Stanford aparc-reduced volume.
# Only cortical + subcortical grey-matter regions are kept as connectome
# nodes; label 1 (white matter) and label 2 (corpus callosum) are used only
# to build the tracking mask, not as regions of interest.
REGION_NAMES = {
    3: "lh-frontalpole", 4: "lh-medialorbitofrontal", 5: "lh-lateralorbitofrontal",
    6: "lh-parsorbitalis", 7: "lh-parstriangularis", 8: "lh-parsopercularis",
    9: "lh-rostralmiddlefrontal", 10: "lh-caudalmiddlefrontal", 11: "lh-superiorfrontal",
    12: "lh-precentral", 13: "lh-paracentral", 14: "lh-insula",
    15: "lh-rostralanteriorcingulate", 16: "lh-caudalanteriorcingulate",
    17: "lh-posteriorcingulate", 18: "lh-isthmuscingulate", 19: "lh-entorhinal",
    20: "lh-fusiform", 21: "lh-parahippocampal", 22: "lh-inferiortemporal",
    23: "lh-temporalpole", 24: "lh-middletemporal", 25: "lh-superiortemporal",
    26: "lh-transversetemporal", 27: "lh-bankssts", 28: "lh-postcentral",
    29: "lh-supramarginal", 30: "lh-inferiorparietal", 31: "lh-superiorparietal",
    32: "lh-precuneus", 33: "lh-cuneus", 34: "lh-lateraloccipital",
    35: "lh-pericalcarine", 36: "lh-lingual",
    37: "lh-caudate", 38: "lh-putamen", 39: "lh-pallidum", 40: "lh-thalamus",
    41: "lh-thalamus-proper", 42: "lh-hippocampus", 43: "lh-amygdala",
    44: "lh-accumbens", 45: "lh-ventralDC",
    46: "rh-frontalpole", 47: "rh-medialorbitofrontal", 48: "rh-lateralorbitofrontal",
    49: "rh-parsorbitalis", 50: "rh-parstriangularis", 51: "rh-parsopercularis",
    52: "rh-rostralmiddlefrontal", 53: "rh-caudalmiddlefrontal", 54: "rh-superiorfrontal",
    55: "rh-precentral", 56: "rh-paracentral", 57: "rh-insula",
    58: "rh-rostralanteriorcingulate", 59: "rh-caudalanteriorcingulate",
    60: "rh-posteriorcingulate", 61: "rh-isthmuscingulate", 62: "rh-entorhinal",
    63: "rh-fusiform", 64: "rh-parahippocampal", 65: "rh-inferiortemporal",
    66: "rh-temporalpole", 67: "rh-middletemporal", 68: "rh-superiortemporal",
    69: "rh-transversetemporal", 70: "rh-bankssts", 71: "rh-postcentral",
    72: "rh-supramarginal", 73: "rh-inferiorparietal", 74: "rh-superiorparietal",
    75: "rh-precuneus", 76: "rh-cuneus", 77: "rh-lateraloccipital",
    78: "rh-pericalcarine", 79: "rh-lingual",
    80: "rh-caudate", 81: "rh-putamen", 82: "rh-pallidum", 83: "rh-thalamus",
    84: "rh-thalamus-proper", 85: "rh-hippocampus", 86: "rh-amygdala",
    87: "rh-accumbens", 88: "rh-ventralDC",
}

print("fetching + loading Stanford HARDI diffusion data and parcellation ...")
img, gtab, labels_img = read_stanford_labels()
data = img.get_fdata()
labels = np.asarray(labels_img.dataobj).astype(int)
affine = img.affine
print("  dwi volume:", data.shape)

white_matter = (labels == 1) | (labels == 2)
print("  white matter voxels:", white_matter.sum())

print("fitting tensor model for FA (tracking stop criterion + seeds) ...")
tenfit = TensorModel(gtab).fit(data, mask=white_matter)
fa = fractional_anisotropy(tenfit.evals)
fa[np.isnan(fa)] = 0

print("estimating fibre response function and fitting CSD model ...")
response, ratio = auto_response_ssst(gtab, data, roi_radii=10, fa_thr=0.7)
csd_model = ConstrainedSphericalDeconvModel(gtab, response)
csd_peaks = peaks_from_model(csd_model, data, default_sphere,
                              relative_peak_threshold=0.5,
                              min_separation_angle=25,
                              mask=white_matter, parallel=False)

print("running deterministic whole-brain tractography ...")
stopping_criterion = ThresholdStoppingCriterion(fa, 0.2)
seeds = utils.seeds_from_mask(white_matter, affine, density=1)
streamlines = Streamlines(
    LocalTracking(csd_peaks, stopping_criterion, seeds, affine, step_size=0.5)
)
print(f"  {len(streamlines)} streamlines generated from {len(seeds)} seeds")

print("building the region-by-region streamline-count matrix ...")
M = utils.connectivity_matrix(streamlines, affine, labels, return_mapping=False)

region_ids = sorted(REGION_NAMES.keys())
connectome = M[np.ix_(region_ids, region_ids)].astype(np.float64)
np.fill_diagonal(connectome, 0)  # a streamline that starts and ends in the same region isn't an edge

names = [REGION_NAMES[i] for i in region_ids]
hemisphere = ["left" if n.startswith("lh-") else "right" for n in names]
region_label = [n.split("-", 1)[1] for n in names]

np.save(f"{OUT}/structural_connectome.npy", connectome)
pd.DataFrame({
    "index": range(len(region_ids)),
    "region": region_label,
    "hemisphere": hemisphere,
    "aparc_label_id": region_ids,
}).to_csv(f"{OUT}/structural_region_labels.csv", index=False)

# a small, plottable sample of streamlines for the notebook's visual
rng = np.random.default_rng(0)
sample_idx = rng.choice(len(streamlines), size=min(300, len(streamlines)), replace=False)
sample = [streamlines[i].astype(np.float32) for i in sample_idx]
np.savez_compressed(f"{OUT}/structural_sample_streamlines.npz",
                     streamlines=np.array(sample, dtype=object), affine=affine)

print("\nsaved:")
print(f"  {OUT}/structural_connectome.npy        {connectome.shape}")
print(f"  {OUT}/structural_region_labels.csv      {len(region_ids)} regions")
print(f"  {OUT}/structural_sample_streamlines.npz  {len(sample)} sample streamlines")
print(f"\ntotal streamline count in matrix: {int(connectome.sum())}")
print(f"nonzero region pairs: {int((connectome > 0).sum())} / {connectome.size}")
