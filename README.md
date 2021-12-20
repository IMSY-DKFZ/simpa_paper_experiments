# README

This repository contains the scripts needed to reproduce all the experiments/examples in the paper: SIMPA: An open-source toolkit for simulation and processing of photoacoustic images. 

## Installation
Please follow the installation instructions of the SIMPA package:
https://github.com/CAMI-DKFZ/simpa#simpa-installation-instructions.

In order to make sure that all experiments run smoothly, all the external tools required by SIMPA have to be installed as well:
https://github.com/CAMI-DKFZ/simpa#external-tools-installation-instructions.
IMPORTANT: You have to build the MCX binary with all custom light sources from this GitHub fork:
https://www.github.com/CAMI-DKFZ/mcx

After the successful installation of SIMPA, this package can be installed similarly:
1. `git clone https://github.com/CAMI-DKFZ/simpa_paper_experiments.git`
2. `cd simpa_paper_experiments`
3. `git pull`

Make sure that you have your preferred virtual environment activated and run in the `simpa_paper_experiments` folder:
1. `pip install -r requirements.txt`
2. `python setup.py develop`

## Run experiments

Make sure that your SIMPA path config file `path_config.env` contains the paths to MATLAB and the MCX binary and it is located according to:
https://github.com/CAMI-DKFZ/simpa#path-management

There are two options for running all the experiments:
1. Execute `bash run_all_experiments.sh` located in `simpa_paper_experiments/experiments`.
2. Run each python script separately in the `simpa_paper_experiments/experiments` subdirectories.
The memory and run time footprint experiment has to be executed separately.
This means that the `memory_footprint.py` and `evaluate_memory_footprint_resutls.py` scripts are not meant to be executed by themselves.
Change your directory to `simpa_paper_experiments/experiments/pa_image_simulation` 
Execute: `bash run_memory_footprint_experiment.sh`