# v1dd_public
This repository contains the code for reproducing the figures of the V1DD manuscript as of October 30 2025.


Author: Arefeh Sherafati (arefesherafati@gmail.com), Naomi Donovan (naomi.donovan@ucsf.edu)

## Dependencies

These set of analyses are built upon a version of the [`allen_v1dd`](https://github.com/AllenInstitute/allen_v1dd/blob/main/README.md) repository that was saved in 2023.  This repository is currently private. You need to request access to allen_v1dd prior to using the current repository. After you gained access to allen_v1dd:

-	Clone the current repository & cd into repository folder:
```
git clone https://github.com/sherafatia/v1dd-public.git
cd v1dd-public
```
-	Create a new conda environment:
```
conda create --name v1dd-public python=3.10
```
- Activate your environment:
```
conda activate v1dd-public
```
- Install poetry version 1.5.1 locally:
```
pip install poetry==1.5.1
```
- Install the requirements using poetry (poetry.lock points to a hash of allen-v1dd back in 2023):
```
poetry install 
```
- You also need to install the AllenSDK repository:
```
pip install allensdk
```

## Data
This repository accesses a local copy of the V1DD files in the Neurodata Without Borders (NWB) format. If you are accessing the data via the Allen Insitute, please see [`allen_v1dd's README`](https://github.com/AllenInstitute/allen_v1dd/blob/main/README.md) for files locations.

## Process locally sparse noise and natural scenes data
To reproduce my analysis for locally sparse noise (lsn) and natural scences (ns set1 with 118 images and ns set 2 with 12 images), follow these steps:
- Make sure you activate your conda environment.
-	Run the following Python script which uses the functions under v1dd-public/utils.py and saves the results in an h5 file under the artifacts directory: lsn_ns_metrics_{tag}.h5. This takes about 3 hours to finish for all mice on my PC. Make sure to define your tag to avoid rewriting. I chose "240410" for the most recent analysis.
```
python analysis-runner/run_lsn_ns_analysis_for_plane.py -a {tag}
```
- Make a csv file for each mouse based on the h5 generated in the previous step: {mouse_cre}_{tag}_lsn_ns_metrics.csv:
```
python analysis-runner/generate_metrics_csv.py --h5-tag {tag}
```
At this step, you are finished with processing the lsn and ns set 1 and set 2 stimuli using my code. You should have one h5 output (e.g., lsn_ns_metrics_240610.h5) and 4 csv files (e.g., slc2_240610_lsn_ns_metrics.csv, 3 more for slc4, slc5, and teto1) under your artifacts directory.

## Merge lsn and ns metrics with other stimuli metrics calculated at the Allen Institute
```
notebooks/merge_lsn_ns_and_other_analysis.ipynb
```
This notebook saves the results in a new csv for each mouse: {mouse_cre}_all_stim_metrics_{tag}.csv (e.g., slc2_all_stim_metrics_240610.csv, and 3 more for slc4, slc5, and teto1).

## Run the notebooks to reproduce the manuscript figures
Now you have all the outputs needed to run the notebooks for reproducing the manuscript figures:
```
v1dd_figure1.ipynb
v1dd_figure2.ipynb
v1dd_figure3.ipynb
v1dd_figure4.ipynb (includes code for supplementary figure 10)
v1dd_figure5.ipynb (includes code for supplementary figures 13, 14)
v1dd_supplementary_figure1.ipynb
v1dd_supplementary_figure2.ipynb
v1dd_supplementary_figure3.ipynb
v1dd_supplementary_figure4.ipynb
v1dd_supplementary_figure5.ipynb
v1dd_supplementary_figure6.ipynb
v1dd_supplementary_figure7.ipynb
v1dd_supplementary_figure8.ipynb
v1dd_supplementary_figure9.ipynb
v1dd_supplementary_figure11.ipynb
v1dd_supplementary_figure12.ipynb
```
