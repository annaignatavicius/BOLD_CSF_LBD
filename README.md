# BOLD-CSF coupling in Lewy body disorders

This repository contains analysis code for quantification of global and regional BOLD-CSF coupling and deidentified source data from Parkinson’s disease, dementia with Lewy bodies and control participants.

The BOLD-CSF coupling method is adapted from [Fultz et al. (2019)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7309589/). Changes in the global BOLD signal are temporally anti-correlated with changes in the CSF inflow signal, providing a measure of the coupling of cerebrovascular dynamics and pulsatile CSF displacement.

## General overview of analytic workflow:

### 1. Preprocessing

Resting state functional MRI data were processed with fMRIPrep (for further details, see [fMRIPrep](https://fmriprep.org/en/stable/) documentation).

### 2. Signal extraction and denoising

The global BOLD (gBOLD) signal was calculated by averaging voxelwise BOLD timeseries across gray matter defined using the [Harvard-Oxford structural atlas](https://neurovault.org/collections/262/). The atlas was transformed from standard MNI space (MNI152NLin6Asym) into each participant’s native functional space using transforms generated during preprocessing.

The CSF inflow signal was extracted from the raw (unprocessed) functional EPI image using a manually delineated mask placed at the most inferior functional slice containing a clearly identifiable CSF inflow signal.

BOLD and CSF time series underwent linear and quadratic detrending, without additional motion-parameter regression, followed by band-pass filtering between 0.01 and 0.10 Hz. Global BOLD and CSF signals were calculated as the spatial mean across voxels within their respective masks and temporally z-scored prior to cross-correlation analysis.

### 3. Global BOLD-CSF coupling

Global BOLD-CSF coupling was quantified as the cross-correlation between the gBOLD and CSF signal across time lags -18 to +18 seconds (+/- 6 TRs).

Group-level peak negative lag was identified and used to quantify participant-level coupling strength. Coupling was also assessed between the negative first-order temporal derivative of gBOLD and CSF signal.

### 4. Regional BOLD-CSF coupling

Voxelwise BOLD signals were averaged within each cortical parcel from the [Schaefer 400-parcel functional atlas](https://github.com/ThomasYeoLab/CBIG/tree/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/MNI), following transformation into participant native functional space.

Parcelwise BOLD–CSF coupling was quantified using the cross-correlation between each parcel BOLD time series and the CSF signal, with coupling defined at the parcel-specific negative cross-correlation peak.

Parcelwise coupling estimates were then averaged according to the Yeo 7 network parcellation and summarized into three hierarchical network levels:

- Unimodal: visual and somatomotor networks
- Attentional: dorsal and ventral attention networks
- Transmodal: frontoparietal and default mode networks

## Analysis code:

### [get_gBOLD_CSF_timeseries.py](https://github.com/annaignatavicius/BOLD_CSF_LBD/blob/main/code/get_gBOLD_CSF_timeseries.py)

Extracts and denoises the global gray-matter BOLD and CSF timeseries.

**Required inputs:**
- Preprocessed resting-state BOLD data
- BOLD reference image
- MNI-to-T1w and BOLD-reference-to-T1w transforms
- Harvard-Oxford cerebral gray-matter atlas in MNI space
- Raw resting-state functional EPI image
- CSF mask

### [BOLD_CSF_coupling.m](https://github.com/annaignatavicius/BOLD_CSF_LBD/blob/main/code/BOLD_CSF_coupling.m)

Calculates global BOLD–CSF cross-correlation functions, participant-level coupling strength, derivative-based coupling, and permutation-based significance testing.

**Required inputs:**
- Z-scored global BOLD timeseries
- Z-scored CSF inflow timeseries

### [get_parcelBOLD.py](https://github.com/annaignatavicius/BOLD_CSF_LBD/blob/main/code/get_parcelBOLD.py)

Transforms the Schaefer-400 atlas into participant functional space and calculates parcelwise cortical BOLD time series.

**Required inputs:**
- Cleaned voxelwise gray-matter BOLD timeseries generated in [get_gBOLD_CSF_timeseries.py](https://github.com/annaignatavicius/BOLD_CSF_LBD/blob/main/code/get_gBOLD_CSF_timeseries.py)
- Subject-space gray-matter mask used to extract the voxelwise BOLD timeseries
- Schaefer 2018 400-parcel, 7-network atlas in MNI space
- BOLD reference image
- MNI-to-T1w and BOLD-reference-to-T1w transforms

### [parcelwise_coupling.m](https://github.com/annaignatavicius/BOLD_CSF_LBD/blob/main/code/parcelwise_coupling.m)

Calculates parcelwise BOLD–CSF coupling.

**Required inputs:**
- Z-scored Parcelwise BOLD timeseries
- Z-scored CSF inflow timeseries

## Source data:

[SourceData_BOLD_CSF_LBD.xlsx](https://github.com/annaignatavicius/BOLD_CSF_LBD/blob/main/data/SourceData_BOLD_CSF_LBD.xlsx) contains deidentified demographic, clinical, cognitive, BOLD-CSF coupling and volumetric data used in associated analysis and generation of tables and figures.

The raw anonymized data are available upon reasonable request.

## Required software and dependencies:

The analysis pipeline uses the following software (see [Key Resource Table](https://github.com/annaignatavicius/BOLD_CSF_LBD/blob/main/KeyResourcesTable.csv)):

- fMRIPrep for functional MRI preprocessing
- Python 3 for signal extraction, denoising, transformation and cortical parcellation
- MATLAB for BOLD-CSF cross-correlation analyses

### MATLAB dependencies:
- Signal Processing Toolbox (xcorr)

### Python dependencies:
- NumPy
- NiBabel
- Nilearn
- SimpleITK
- SciPy

## References

### BOLD-CSF coupling:

Fultz, N. E., Bonmassar, G., Setsompop, K., Stickgold, R. A., Rosen, B. R., Polimeni, J. R., & Lewis, L. D. (2019). Coupled electrophysiological, hemodynamic, and cerebrospinal fluid oscillations in human sleep. *Science. 366*(6465), 628–631. [https://doi.org/10.1126/science.aax5440](https://doi.org/10.1126/science.aax5440)

### Preprocessing:

Esteban, O., Markiewicz, C. J., Blair, R. W., Moodie, C. A., Isik, A. I., Erramuzpe, A., Kent, J. D., Goncalves, M., DuPre, E., Snyder, M., Oya, H., Ghosh, S. S., Wright, J., Durnez, J., Poldrack, R. A., & Gorgolewski, K. J. (2019). fMRIPrep: a robust preprocessing pipeline for functional MRI. *Nature methods, 16*(1), 111–116. [https://doi.org/10.1038/s41592-018-0235-4](https://doi.org/10.1038/s41592-018-0235-4)

### Atlases:

Desikan, R. S., Ségonne, F., Fischl, B., Quinn, B. T., Dickerson, B. C., Blacker, D., Buckner, R. L., Dale, A. M., Maguire, R. P., Hyman, B. T., Albert, M. S., & Killiany, R. J. (2006). An automated labeling system for subdividing the human cerebral cortex on MRI scans into gyral based regions of interest. *NeuroImage*, 31(3), 968–980. [https://doi.org/10.1016/j.neuroimage.2006.01.021](https://doi.org/10.1016/j.neuroimage.2006.01.021)

Schaefer, A., Kong, R., Gordon, E. M., Laumann, T. O., Zuo, X. N., Holmes, A. J., Eickhoff, S. B., & Yeo, B. T. T. (2018). Local-Global Parcellation of the Human Cerebral Cortex from Intrinsic Functional Connectivity MRI. *Cerebral cortex, 28*(9), 3095–3114. [https://doi.org/10.1093/cercor/bhx179](https://doi.org/10.1093/cercor/bhx179)

Yeo, B. T., Krienen, F. M., Sepulcre, J., Sabuncu, M. R., Lashkari, D., Hollinshead, M., Roffman, J. L., Smoller, J. W., Zöllei, L., Polimeni, J. R., Fischl, B., Liu, H., & Buckner, R. L. (2011). The organization of the human cerebral cortex estimated by intrinsic functional connectivity. *Journal of Neurophysiology*, 106(3), 1125–1165. [https://doi.org/10.1152/jn.00338.2011](https://doi.org/10.1152/jn.00338.2011)

