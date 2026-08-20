"""Build the cached functional connectome used by notebook 8's structure-function
comparison, already reduced and ordered to match the structural connectome from
notebook 7's `structural_connectome.npy`.

Important honesty note, also repeated on screen in notebook 8: this functional
data comes from ONE ADULT PARTICIPANT in nilearn's open "development fMRI"
dataset. The structural connectome in notebook 7 comes from a DIFFERENT person
(DIPY's Stanford HARDI subject). These are not the same brain. We are
demonstrating the comparison method, not claiming a real within-subject
structure-function result — a genuine version of this analysis needs diffusion
and resting-state data acquired from the same person.

There is a second mismatch this script has to resolve: notebook 7's structural
regions come from a Desikan-Killiany-style parcellation (`aparc-reduced`,
shipped registered to the Stanford subject's own anatomy), while functional
connectivity here is extracted with the AAL3 atlas (a standard, separately
built MNI-space anatomical atlas) because there is no off-the-shelf atlas that
is both a labelled volume in MNI space AND already registered to the Stanford
subject. AAL3 and Desikan-Killiany carve the same rough anatomy differently in
detail, so region correspondence below is by anatomical NAME, not by shared
voxels — a curated, approximate lookup table, not an exact registration. Only
region pairs with a confident name-level match are kept.

Run once:  python scripts/build_functional_for_comparison.py
Takes about 20-30 seconds (one small fMRI download + one atlas fetch).
"""

import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from nilearn import datasets
from nilearn.maskers import NiftiLabelsMasker

OUT = "data"
os.makedirs(OUT, exist_ok=True)

# DK region name (as saved in structural_region_labels.csv, without lh-/rh-)
# -> one or more AAL3 base region names (hemisphere suffix _L/_R added below).
# Curated by anatomical name; entries with no confident AAL match are left out
# on purpose (frontal pole, banks of the STS, entorhinal cortex, isthmus
# cingulate, the duplicate FreeSurfer "thalamus" label, ventral DC, and the
# caudal/rostral middle-frontal split, which AAL does not distinguish).
DK_TO_AAL = {
    "medialorbitofrontal": ["Frontal_Med_Orb"],
    "lateralorbitofrontal": ["OFClat"],
    "parsorbitalis": ["Frontal_Inf_Orb_2"],
    "parstriangularis": ["Frontal_Inf_Tri"],
    "parsopercularis": ["Frontal_Inf_Oper"],
    "rostralmiddlefrontal": ["Frontal_Mid_2"],
    "superiorfrontal": ["Frontal_Sup_2"],
    "precentral": ["Precentral"],
    "paracentral": ["Paracentral_Lobule"],
    "insula": ["Insula"],
    "rostralanteriorcingulate": ["ACC_pre"],
    "caudalanteriorcingulate": ["ACC_sup"],
    "posteriorcingulate": ["Cingulate_Post"],
    "fusiform": ["Fusiform"],
    "parahippocampal": ["ParaHippocampal"],
    "inferiortemporal": ["Temporal_Inf"],
    "temporalpole": ["Temporal_Pole_Sup"],
    "middletemporal": ["Temporal_Mid"],
    "superiortemporal": ["Temporal_Sup"],
    "transversetemporal": ["Heschl"],
    "postcentral": ["Postcentral"],
    "supramarginal": ["SupraMarginal"],
    "inferiorparietal": ["Parietal_Inf"],
    "superiorparietal": ["Parietal_Sup"],
    "precuneus": ["Precuneus"],
    "cuneus": ["Cuneus"],
    "lateraloccipital": ["Occipital_Mid"],
    "pericalcarine": ["Calcarine"],
    "lingual": ["Lingual"],
    "caudate": ["Caudate"],
    "putamen": ["Putamen"],
    "pallidum": ["Pallidum"],
    "thalamus-proper": ["Thal_AV", "Thal_LP", "Thal_VA", "Thal_VL", "Thal_VPL",
                         "Thal_IL", "Thal_Re", "Thal_MDm", "Thal_MDl", "Thal_LGN",
                         "Thal_MGN", "Thal_PuI", "Thal_PuM", "Thal_PuA", "Thal_PuL"],
    "hippocampus": ["Hippocampus"],
    "amygdala": ["Amygdala"],
    "accumbens": ["N_Acc"],
}

print("loading the structural connectome from notebook 7 ...")
structural = np.load(f"{OUT}/structural_connectome.npy")
struct_labels = pd.read_csv(f"{OUT}/structural_region_labels.csv")

print("fetching AAL3 atlas ...")
aal = datasets.fetch_atlas_aal()
aal_name_to_index = dict(zip(aal.labels, [int(i) for i in aal.indices]))

print("fetching one adult participant from nilearn's development fMRI dataset ...")
dev = datasets.fetch_development_fmri(n_subjects=1, age_group="adult")
subject_id = pd.DataFrame(dev.phenotypic)["participant_id"].iloc[0]
print(f"  functional subject: {subject_id}  (NOT the Stanford diffusion subject)")

print("extracting AAL3 region time series ...")
masker = NiftiLabelsMasker(labels_img=aal.maps, standardize="zscore_sample", verbose=0)
ts = masker.fit_transform(dev.func[0], confounds=dev.confounds[0])
retained_aal_index = [int(i) for i in masker.labels_]
print(f"  time series: {ts.shape[0]} timepoints x {ts.shape[1]} AAL regions retained")

# build the matched region list: every (dk_region, hemisphere) with a
# confident AAL name AND whose AAL region(s) survived masking
matched_rows = []
for _, row in struct_labels.iterrows():
    dk_region = row["region"]
    if dk_region not in DK_TO_AAL:
        continue
    suffix = "_L" if row["hemisphere"] == "left" else "_R"
    aal_bases = DK_TO_AAL[dk_region]
    aal_indices = []
    ok = True
    for base in aal_bases:
        name = base + suffix
        idx = aal_name_to_index.get(name)
        if idx is None or idx not in retained_aal_index:
            ok = False
            break
        aal_indices.append(retained_aal_index.index(idx))
    if ok:
        matched_rows.append({
            "structural_index": int(row["index"]),
            "dk_region": dk_region,
            "hemisphere": row["hemisphere"],
            "aal_columns": aal_indices,
        })

print(f"matched {len(matched_rows)} / {len(struct_labels)} structural regions to AAL3 by name")

# functional time series per matched region (summed across AAL sub-parcels, e.g. thalamus nuclei)
matched_ts = np.stack([ts[:, cols].mean(axis=1) for cols in (r["aal_columns"] for r in matched_rows)], axis=1)
functional = np.corrcoef(matched_ts.T)
np.fill_diagonal(functional, 0)

structural_matched = structural[np.ix_([r["structural_index"] for r in matched_rows],
                                        [r["structural_index"] for r in matched_rows])]

out_labels = pd.DataFrame([{"dk_region": r["dk_region"], "hemisphere": r["hemisphere"]}
                            for r in matched_rows])

np.save(f"{OUT}/structural_connectome_matched.npy", structural_matched)
np.save(f"{OUT}/functional_connectome_matched.npy", functional)
out_labels.to_csv(f"{OUT}/comparison_region_labels.csv", index=False)
with open(f"{OUT}/comparison_functional_subject.txt", "w") as f:
    f.write(f"{subject_id}\n")

print("\nsaved:")
print(f"  {OUT}/structural_connectome_matched.npy   {structural_matched.shape}")
print(f"  {OUT}/functional_connectome_matched.npy   {functional.shape}")
print(f"  {OUT}/comparison_region_labels.csv         {len(out_labels)} regions")
print(f"  {OUT}/comparison_functional_subject.txt    '{subject_id}'")
