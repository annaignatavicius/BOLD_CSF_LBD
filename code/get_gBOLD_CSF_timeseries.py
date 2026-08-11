#!/usr/bin/env python3
import os
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

from pathlib import Path
import tempfile
import shutil

import nibabel as nib
import numpy as np
import SimpleITK as sitk
from nilearn.input_data import NiftiMasker
from nilearn import signal as nl_signal

# Set parameters/settings

# fMRIPrep derivatives root -> change for each group
IN_ROOT = Path(
    "/Volumes/research-data/PRJ-Ach_VH/test_fmriprep_output/derivatives_noSDC/DLB/DLB"
)


CSF_ROOT = Path(
    "/Volumes/DYNABOOK/BOLD_CSF/bids/dlb"
)

TASK_NAME = "rest"

OUT_ROOT = Path(
    "/Volumes/DYNABOOK/BOLD_CSF/DLB_timeseries_atlas_new"
)

# Template Schaefer-400/7-network atlas in MNI space
ATLAS_MNI = Path(
    "/Volumes/DYNABOOK/BOLD_CSF/HarvardOxford_cerebrumGM_thr25_2mm_mask.nii.gz"
)


# Denoising: linear + quadratic trends, no motion regression, 0.01-0.10 Hz band-pass
HIGH_PASS = 0.01
LOW_PASS = 0.10

BOLD_PAT = f"*_task-{TASK_NAME}_space-T1w_desc-preproc_bold.nii.gz"
BOLDREF_PAT = f"*_task-{TASK_NAME}_space-T1w_boldref.nii.gz"
COREG_PAT = f"*_task-{TASK_NAME}_from-boldref_to-T1w_mode-image_desc-coreg_xfm.txt"
MNI_TO_T1_PAT = "*_from-MNI152NLin6Asym_to-T1w_mode-image_xfm.h5"


# Functions

def read_transform_with_retry(path: Path):
    """Read transform; retry from a local copy if the external drive causes trouble."""
    try:
        return sitk.ReadTransform(str(path))
    except Exception:
        tmpdir = Path(tempfile.mkdtemp(prefix="xfm_"))
        tmp = tmpdir / path.name
        shutil.copy2(path, tmp)
        return sitk.ReadTransform(str(tmp))


def find_runs(root: Path):
    """Find rest BOLD runs with the transforms needed for atlas warping."""
    runs = []
    bold_files = sorted(p for p in root.rglob(BOLD_PAT) if not p.name.startswith("._"))

    print(f"[scan] Found {len(bold_files)} BOLD files")

    for bold in bold_files:
        func_dir = bold.parent
        sub = next((p for p in bold.parts if p.startswith("sub-")), "sub-UNK")
        ses = next((p for p in bold.parts if p.startswith("ses-")), None)

        boldref = next(func_dir.glob(BOLDREF_PAT), None)
        coreg = next(func_dir.glob(COREG_PAT), None)

        # session-level anat, or subject-level anat if there is no session transform
        anat_dir = bold.parents[1] / "anat"
        h5 = next(anat_dir.glob(MNI_TO_T1_PAT), None) if anat_dir.exists() else None

        if h5 is None and ses is not None:
            anat_dir = bold.parents[2] / "anat"
            h5 = next(anat_dir.glob(MNI_TO_T1_PAT), None) if anat_dir.exists() else None

        if boldref is None or coreg is None or h5 is None:
            print(f"[skip] {sub} {ses or 'no-ses'}: missing atlas-warp file(s)")
            continue

        runs.append({
            "sub": sub,
            "ses": ses,
            "bold": bold,
            "boldref": boldref,
            "coreg": coreg,
            "h5": h5,
        })

    print(f"[scan] {len(runs)} usable runs")
    return runs


def warp_atlas_to_bold(atlas_mni: Path,
                       boldref: Path,
                       h5_mni_to_t1: Path,
                       coreg_boldref_to_t1: Path,
                       out_mask: Path):
    """Warp the fixed MNI GM atlas to the subject's space-T1w BOLD grid."""
    atlas = sitk.ReadImage(str(atlas_mni))
    bref = sitk.ReadImage(str(boldref))

    mni_to_t1 = read_transform_with_retry(h5_mni_to_t1)
    boldref_to_t1 = sitk.ReadTransform(str(coreg_boldref_to_t1))
    t1_to_boldref = boldref_to_t1.GetInverse()

    mni_to_boldref = sitk.CompositeTransform([t1_to_boldref, mni_to_t1])

    warped = sitk.Resample(
        atlas,
        bref,
        mni_to_boldref,
        sitk.sitkNearestNeighbor,
        0,
        sitk.sitkUInt8,
    )

    arr = (sitk.GetArrayFromImage(warped) > 0).astype(np.uint8)
    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(warped)

    out_mask.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(out, str(out_mask))
    return out_mask


def denoise_timeseries(voxel_ts: np.ndarray, tr: float):
    """Linear + quadratic detrending and 0.01-0.10 Hz filtering."""
    n_tp = voxel_ts.shape[0]
    t = np.linspace(-1.0, 1.0, n_tp)
    trends = np.column_stack([
        t,
        t**2 - np.mean(t**2),
    ])

    return nl_signal.clean(
        voxel_ts,
        confounds=trends,
        detrend=False,
        standardize=False,
        high_pass=HIGH_PASS,
        low_pass=LOW_PASS,
        t_r=tr,
    )


def extract_clean_timeseries(bold_path: Path,
                             mask_path: Path,
                             out_dir: Path,
                             tag: str):
    """Extract mask voxels, denoise them, and save voxelwise + mean clean signals."""
    bold_img = nib.load(str(bold_path))
    mask_img = nib.load(str(mask_path))

    if bold_img.shape[:3] != mask_img.shape[:3]:
        raise RuntimeError(
            f"{tag}: BOLD shape {bold_img.shape[:3]} != mask shape {mask_img.shape[:3]}"
        )

    if not np.allclose(bold_img.affine, mask_img.affine, atol=1e-4):
        raise RuntimeError(f"{tag}: BOLD and mask affines do not match")

    tr = float(bold_img.header.get_zooms()[3])

    masker = NiftiMasker(
        mask_img=str(mask_path),
        smoothing_fwhm=None,
        detrend=False,
        standardize=False,
        ensure_finite=True,
    )

    voxel_ts = masker.fit_transform(str(bold_path))
    if voxel_ts.shape[1] == 0:
        raise RuntimeError(f"{tag}: mask contains zero voxels")

    voxel_clean = denoise_timeseries(voxel_ts, tr)
    mean_clean = voxel_clean.mean(axis=1)

    out_dir.mkdir(parents=True, exist_ok=True)
    base = bold_path.name.removesuffix(".nii.gz")

    np.save(
        out_dir / f"{base}_{tag}_vBOLD_clean.npy",
        voxel_clean.astype(np.float32),
    )
    np.savetxt(
        out_dir / f"{base}_{tag}_gBOLD_clean.tsv",
        mean_clean,
        fmt="%.8f",
    )

    print(f"[{tag}] T={voxel_clean.shape[0]}, V={voxel_clean.shape[1]}")


def find_csf_paths(sub: str, ses: str | None):
    """Find the existing raw rest BOLD + CSF mask."""
    dirs = []
    if ses:
        dirs.append(CSF_ROOT / sub / ses / "func")
    dirs.append(CSF_ROOT / sub / "func")

    for func_dir in dirs:
        if not func_dir.exists():
            continue

        bold = next(
            (p for p in func_dir.glob(f"*_task-{TASK_NAME}_bold.nii.gz")
             if not p.name.startswith("._")),
            None,
        )
        mask = next(
            (p for p in func_dir.glob(f"*_task-{TASK_NAME}_bold_CSF_mask.nii.gz")
             if not p.name.startswith("._")),
            None,
        )

        if bold is not None and mask is not None:
            return bold, mask

    return None, None


def main():
    if not IN_ROOT.exists():
        raise FileNotFoundError(f"Input root not found: {IN_ROOT}")
    if not ATLAS_MNI.exists():
        raise FileNotFoundError(f"Atlas not found: {ATLAS_MNI}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    runs = find_runs(IN_ROOT)

    for i, run in enumerate(runs, 1):
        sub = run["sub"]
        ses = run["ses"]
        ses_out = ses or "ses-NA"

        print(f"\n[run] {i}/{len(runs)}  {sub}  {ses_out}")

        out_dir = OUT_ROOT / sub / ses_out / "func"
        out_dir.mkdir(parents=True, exist_ok=True)

        base = run["bold"].name.removesuffix(
            "_space-T1w_desc-preproc_bold.nii.gz"
        )
        gm_mask = out_dir / f"{base}_GM_atlas_epi.nii.gz"

        
        if not gm_mask.exists():
            try:
                warp_atlas_to_bold(
                    ATLAS_MNI,
                    run["boldref"],
                    run["h5"],
                    run["coreg"],
                    gm_mask,
                )
                print(f"[GM] wrote atlas mask: {gm_mask.name}")
            except Exception as e:
                print(f"[skip] {sub} {ses_out}: atlas warp failed: {e}")
                continue
        else:
            print(f"[GM] atlas mask exists: {gm_mask.name}")

        try:
            extract_clean_timeseries(run["bold"], gm_mask, out_dir, "GM")
        except Exception as e:
            print(f"[skip] {sub} {ses_out}: GM extraction failed: {e}")
            continue

        # CSF
        csf_bold, csf_mask = find_csf_paths(sub, ses)
        if csf_bold is None or csf_mask is None:
            print(f"[CSF] no BOLD/mask found for {sub} {ses_out}")
            continue

        try:
            extract_clean_timeseries(csf_bold, csf_mask, out_dir, "CSF")
        except Exception as e:
            print(f"[CSF] {sub} {ses_out}: extraction failed: {e}")

    print("\n[done]")


if __name__ == "__main__":
    main()
