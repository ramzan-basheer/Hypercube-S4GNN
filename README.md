# Hypercube-S4GNN: A Multi-Edge Graph Approach Using State Space Models on Multivariate EEG for Seizure Detection

Ramzan Basheer, A. H. Daraie, Deepak Mishra 
Published in International Workshop on Machine learning for Signal Processing, 22 September 2024, London, UK

http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10734832

DOI: 10.1109/MLSP58920.2024.10734832

---
This project is adapted from the works in https://github.com/tsy935/graphs4mer/tree/main. We create a multigraph based on various functional connectivity.

---
## Setup
This codebase requries python ≥ 3.9, pytorch ≥ 1.12.0, and pyg installed. Please refer to [PyTorch installation](https://pytorch.org/) and [PyG installation](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html). Other dependencies are included in `requirements.txt` and can be installed via `pip install -r requirements.txt`

---
## Datasets
### TUSZ
The TUSZ dataset is publicly available and can be accessed from https://isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml after filling out the data request form. We use TUSZ v1.5.2 in this study.
#### TUSZ data preprocessing
First, we resample all EEG signals in TUSZ to 200 Hz. To do so, run:
```
python data/preprocess/resample_tuh.py --raw_edf_dir {dir-to-tusz-edf-files} --save_dir {dir-to-resampled-signals}
```
---
## Model Training
`scripts` folder shows examples to train S4GNN. 
To train the model on the TUSZ dataset, specify `<dir-to-resampled-signals>`, `<preproc-save-dir>`, and `<your-save-dir>` in `scripts/run_tuh.sh`, then run the following:
```
bash ./scripts/run_tuh.sh
```
Note that the first time when you run this script, it will first preprocess the resampled signals by sliding a 60-s window without overlaps and save the 60-s EEG clips and seizure/non-seizure labels in PyG data object in `<preproc-save-dir>`.

