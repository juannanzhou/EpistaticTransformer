Run Script CLI
================

What this does
---------------
Runs the main training/evaluation pipeline. It prepares data splits, fits models (linear and transformer), and writes results and logs to an output folder.

Required environment
------------------
run `conda env create -f environment.yml -n epistatic_transformer_env` to create the conda environment 

Required arguments
------------------
- `--device` : device string for PyTorch (e.g. `cpu` or `cuda:0`)
- `--data_name` : name of the input CSV (without `.csv`) located in `../Data/Data_prepared/`
- `--prefix` : short prefix for the study (used in the output folder name)
- `--train_percent` : percentage of data to use for training (as a number, e.g. `10`)

Input
-----
Place your data file at `../Data/Data_prepared/<data_name>.csv` (relative to `run_scripts`).

Example
-------
python run_script-CLI.py --device cuda:0 --data_name Pokusaeva_2019_S1 --prefix run1 --train_percent 20

Where results are saved
----------------------
Results are written under `../output/` (relative to the repo root) to a folder named:

`<data_name>_<prefix>_<train_percent>%_rep_<n>`

The folder contains `summary.txt`, `output.txt` (log), `R2s.csv`, `train_list.pkl`, `val_list.pkl`, `test_list.pkl`, copies of `models.py` and the run script, and other result files.

Notes
-----
- Other optional flags exist in the script (seed, preset parameter files, prespecified splits, etc.). Only the four required arguments above are needed for a default run.
