#!/usr/bin/env python3
import os
from pathlib import Path

import numpy as np
import nibabel as nib
import SimpleITK as sitk
from scipy.io import savemat

# Set parameters/settings

# Template Schaefer-400/7-network atlas in MNI space
ATLAS_MNI = Path(
    "/Volumes/research-data/PRJ-Ach_VH/tpl-MNI152NLin6Asym_res-02_atlas-Schaefer2018_desc-400Parcels7Networks_dseg.nii.gz"
)

# fMRIPrep derivatives root 
DERIV_ROOT = Path("/Volumes/research-data/PRJ-Ach_VH/test_fmriprep_output/derivatives_noSDC")

# Timeseries roots for each group 
GROUP_NAME      = "PD"  # change for each run
TIMESERIES_ROOT = Path("/Volumes/DYNABOOK/BOLD_CSF/PD_timeseries_atlas") # change for each run

TASK_NAME = "rest"

# File name patterns 
GM_VBOLD_SUFFIX   = "_task-rest_space-T1w_desc-preproc_bold_GM_vBOLD_clean.npy" # voxelwise BOLD timeseries
GM_MASK_PATTERN   = "*_GM_atlas_epi.nii.gz"  # must match the mask used to create GM_vBOLD_*.npy

# Save parcel time series (T x 400) and voxel counts per parcel
MIN_VOXELS_PER_PARCEL = 1

# Functions


def read_transform_with_retry(h5_path: Path, label: str):
    """Robustly read a SimpleITK transform, with a local copy fallback."""
    try:
        return sitk.ReadTransform(str(h5_path))
    except Exception as e1:
        print(f"[warn] ReadTransform failed on {label}: {h5_path.name} ({e1})")
        import tempfile, shutil
        tmpdir = Path(tempfile.mkdtemp(prefix="xfm_"))
        tmp = tmpdir / h5_path.name
        try:
            shutil.copy2(str(h5_path), str(tmp))
            print(f"[retry] copied to {tmp}; retrying read…")
            return sitk.ReadTransform(str(tmp))
        except Exception as e2:
            print(f"[fail] cannot read transform after local copy: {e2}")
            raise


def warp_schaefer_to_bold(atlas_mni_path: Path,
                          boldref_path: Path,
                          h5_mni_to_t1: Path,
                          coreg_bref_to_t1: Path,
                          out_dseg_path: Path) -> Path:
    """
    Warp Schaefer atlas from MNI to subject BOLD (space-T1w boldref) using
    MNI->T1w (h5) and boldref->T1w (txt) transforms. Nearest-neighbour
    resampling.
    """
    print(f"[warp] atlas:   {atlas_mni_path}")
    print(f"[warp] boldref: {boldref_path}")
    print(f"[warp] h5:      {h5_mni_to_t1}")
    print(f"[warp] coreg:   {coreg_bref_to_t1}")

    atlas_mni = sitk.ReadImage(str(atlas_mni_path))
    bref      = sitk.ReadImage(str(boldref_path))

    t_mni_to_t1  = read_transform_with_retry(h5_mni_to_t1, "MNI->T1")
    t_bref_to_t1 = sitk.ReadTransform(str(coreg_bref_to_t1))
    t_t1_to_bref = t_bref_to_t1.GetInverse()

    # Compose MNI->T1 then T1->boldref = MNI->boldref
    t_mni_to_bref = sitk.CompositeTransform([t_t1_to_bref, t_mni_to_t1])

    atlas_epi = sitk.Resample(
        atlas_mni,
        bref,
        t_mni_to_bref,
        sitk.sitkNearestNeighbor,
        0,            # background = 0 (no label)
        sitk.sitkUInt16
    )

    out_dseg_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(atlas_epi, str(out_dseg_path))
    print(f"[warp] wrote subject-space atlas to: {out_dseg_path}")
    return out_dseg_path


def build_parcel_timeseries(func_dir: Path, atlas_dseg_path: Path, n_parcels: int = 400):
    """
    Build cleaned and z-scored parcel BOLD time series (T x 400) from the
    cleaned voxelwise GM BOLD signal, subject-space Schaefer atlas, and the
    exact FOV-intersected GM mask used for voxel extraction.

    Outputs:
      - *_parcelBOLD_clean.npy : cleaned parcel means (T x 400)
      - *_parcelBOLD_z.npy     : temporally z-scored parcel means (T x 400)
      - *_parcelBOLD_z.mat     : same z-scored data as MATLAB variable parcelBOLD
      - *_parcel_nvox.tsv      : number of contributing GM voxels per parcel
    """
    gm_vbold_files = sorted(func_dir.glob(f"*{GM_VBOLD_SUFFIX}"))
    if len(gm_vbold_files) == 0:
        print(f"[skip] no GM_vBOLD_clean file in {func_dir}")
        return
    if len(gm_vbold_files) > 1:
        print(f"[warn] multiple GM_vBOLD files; using {gm_vbold_files[0].name}")
    gm_vbold_path = gm_vbold_files[0]

    gm_mask_files = sorted(func_dir.glob(GM_MASK_PATTERN))
    if len(gm_mask_files) == 0:
        print(f"[skip] no GM mask matching {GM_MASK_PATTERN} in {func_dir}")
        return
    if len(gm_mask_files) > 1:
        print(f"[warn] multiple GM masks; using {gm_mask_files[0].name}")
    gm_mask_path = gm_mask_files[0]

    print(f"[parcel] GM vBOLD: {gm_vbold_path.name}")
    print(f"[parcel] GM mask:  {gm_mask_path.name}")
    print(f"[parcel] Atlas:    {atlas_dseg_path.name}")

    # Cleaned voxelwise GM BOLD (T x Vgm).
    gm_vbold = np.load(gm_vbold_path)
    if gm_vbold.ndim != 2:
        raise RuntimeError(
            f"Expected GM vBOLD to be 2D (T x Vgm). Got {gm_vbold.shape}"
        )
    T, Vgm = gm_vbold.shape

    atlas_img = nib.load(str(atlas_dseg_path))
    atlas_data = atlas_img.get_fdata().astype(np.int32)

    gm_mask_img = nib.load(str(gm_mask_path))
    gm_mask_data = gm_mask_img.get_fdata().astype(bool)

    if atlas_data.shape != gm_mask_data.shape:
        raise RuntimeError(
            "Atlas dseg and GM mask shapes differ. Check spaces/resolutions."
        )
    if not np.allclose(atlas_img.affine, gm_mask_img.affine, atol=1e-4):
        raise RuntimeError(
            "Atlas dseg and GM mask affines differ. Check spaces/resolutions."
        )

    # Apply the exact GM mask used by NiftiMasker so each vBOLD column gets
    # the corresponding Schaefer parcel label.
    labels_1d = atlas_data[gm_mask_data]
    if labels_1d.shape[0] != Vgm:
        raise RuntimeError(
            f"Number of GM mask voxels ({labels_1d.shape[0]}) != "
            f"vBOLD columns ({Vgm}).\n"
            "The GM mask must be the exact mask used to extract "
            "GM_vBOLD_clean.npy."
        )

    P = n_parcels
    parcel_ts_clean = np.full((T, P), np.nan, dtype=np.float64)
    nvox = np.zeros(P, dtype=np.int32)

    for pid in range(1, P + 1):
        vox = labels_1d == pid
        n = int(np.sum(vox))
        nvox[pid - 1] = n
        if n < MIN_VOXELS_PER_PARCEL:
            continue

        # Spatial mean of the already-denoised voxelwise BOLD signal.
        parcel_ts_clean[:, pid - 1] = gm_vbold[:, vox].mean(axis=1)

    # Temporal z-score is applied AFTER averaging voxels within each parcel.
    mu = np.nanmean(parcel_ts_clean, axis=0, keepdims=True)
    sd = np.nanstd(parcel_ts_clean, axis=0, ddof=1, keepdims=True)
    sd[(~np.isfinite(sd)) | (sd == 0)] = np.nan
    parcel_ts_z = (parcel_ts_clean - mu) / sd

    base = gm_vbold_path.name.replace(GM_VBOLD_SUFFIX, "")

    out_clean = func_dir / f"{base}_parcelBOLD_clean.npy"
    out_z_npy = func_dir / f"{base}_parcelBOLD_z.npy"
    out_z_mat = func_dir / f"{base}_parcelBOLD_z.mat"
    out_nvox = func_dir / f"{base}_parcel_nvox.tsv"

    np.save(out_clean, parcel_ts_clean.astype(np.float32))
    np.save(out_z_npy, parcel_ts_z.astype(np.float32))

    # parcelwise_coupling.m expects *_parcelBOLD_z.mat with variable parcelBOLD.
    savemat(
        out_z_mat,
        {"parcelBOLD": parcel_ts_z.astype(np.float32)},
        do_compression=True,
    )

    np.savetxt(out_nvox, nvox, fmt="%d")

    print(f"[parcel] wrote {out_clean.name}")
    print(f"[parcel] wrote {out_z_npy.name}")
    print(f"[parcel] wrote {out_z_mat.name} [variable: parcelBOLD]")
    print(f"[parcel] wrote {out_nvox.name}")


def main():
    if not ATLAS_MNI.exists():
        raise FileNotFoundError(f"Template atlas not found: {ATLAS_MNI}")

    group_deriv_root = DERIV_ROOT / GROUP_NAME
    if not group_deriv_root.exists():
        raise FileNotFoundError(f"Derivatives group folder not found: {group_deriv_root}")

    # subjects in root
    subs = [d for d in TIMESERIES_ROOT.iterdir()
            if d.is_dir() and d.name.startswith("sub-")]

    print(f"[scan] Found {len(subs)} subjects in {TIMESERIES_ROOT}")

    for sub_dir in subs:
        sub = sub_dir.name
        print(f"\n {sub}")

        # session folders under timeseries root
        ses_dirs = [d for d in sub_dir.iterdir()
                    if d.is_dir() and d.name.startswith("ses-")]
        if not ses_dirs:
            ses_dirs = [sub_dir]

        for ses_dir in ses_dirs:
            ses_label = None if ses_dir is sub_dir else ses_dir.name

            func_dir = ses_dir / "func"
            if not func_dir.exists():
                print(f"[skip] no func dir in {ses_dir}")
                continue

            print(f"[session] {sub} | {ses_label or 'no-ses'} | func: {func_dir}")

            # derivatives subject dir
            deriv_sub_dir = group_deriv_root / sub
            if not deriv_sub_dir.exists():
                print(f"[skip] derivatives folder missing: {deriv_sub_dir}")
                continue

            # find boldref + coreg (boldref->T1w) in derivatives
            if ses_label and ses_label != "ses-NA":
                cand_func_dirs = [deriv_sub_dir / ses_label / "func"]
                cand_func_dirs = [d for d in cand_func_dirs if d.exists()]
            else:
                cand_func_dirs = [d for d in deriv_sub_dir.rglob("func")]

            boldref = None
            coreg = None
            for fdir in cand_func_dirs:
                cand_bref = list(fdir.glob(f"{sub}_*task-{TASK_NAME}_space-T1w_boldref.nii.gz"))
                if not cand_bref:
                    continue
                cand_coreg = list(fdir.glob(
                    f"{sub}_*task-{TASK_NAME}_from-boldref_to-T1w_mode-image_desc-coreg_xfm.txt"
                ))
                if not cand_coreg:
                    continue
                boldref = cand_bref[0]
                coreg = cand_coreg[0]
                break

            if boldref is None or coreg is None:
                print(f"[skip] missing boldref/coreg for {sub} {ses_label or ''} in derivatives.")
                continue

            # anat dir + MNI -> T1 h5 transform
            if ses_label and ses_label != "ses-NA":
                anat_dir = deriv_sub_dir / ses_label / "anat"
                if not anat_dir.exists():
                    anat_dir = deriv_sub_dir / "anat"
            else:
                anat_dir = deriv_sub_dir / "anat"

            h5_cand = list(anat_dir.glob(f"{sub}_*from-MNI152NLin6Asym_to-T1w_mode-image_xfm.h5"))
            if not h5_cand:
                print(f"[skip] missing MNI->T1w h5 for {sub} in {anat_dir}")
                continue
            h5 = h5_cand[0]

            # subject-space atlas path in the TIMESERIES func_dir
            atlas_dseg_path = func_dir / f"{sub}_space-T1w_atlas-Schaefer2018_desc-400Parcels7Networks_dseg.nii.gz"

            if not atlas_dseg_path.exists():
                try:
                    warp_schaefer_to_bold(
                        atlas_mni_path=ATLAS_MNI,
                        boldref_path=boldref,
                        h5_mni_to_t1=h5,
                        coreg_bref_to_t1=coreg,
                        out_dseg_path=atlas_dseg_path,
                    )
                except Exception as e:
                    print(f"[skip] warping failed for {sub} {ses_label or ''}: {e}")
                    continue
            else:
                print(f"[warp] subject-space atlas already exists: {atlas_dseg_path.name}")

            # build parcel timeseries (T x 400 vectors, saved per subject)
            try:
                build_parcel_timeseries(func_dir, atlas_dseg_path, n_parcels=400)
            except Exception as e:
                print(f"[error] building parcel TS failed for {sub} {ses_label or ''}: {e}")
                continue

    print("\n[done] All subjects + sessions processed.")


if __name__ == "__main__":
    main()
