# README

This repository contains the scripts needed to reproduce all the experiments/examples in the paper: SIMPA: An open-source toolkit for simulation and processing of photoacoustic images. 

## Installation
Please follow the installation instructions of the SIMPA package:
https://github.com/CAMI-DKFZ/simpa#simpa-install-instructions.

In order to make sure that all experiments run smoothly, all the external tools required by SIMPA have to be installed as well:
https://github.com/CAMI-DKFZ/simpa#external-tools-installation-instructions.
IMPORTANT: You have to build the MCX binary with all custom light sources from this GitHub fork:
https://www.github.com/jgroehl/mcx

## Run experiments

Make sure that your SIMPA path config file `path_config.env` contains the paths to MATLAB and the MCX binary and it is located according to:
https://github.com/CAMI-DKFZ/simpa#path-management

There are two options for running all the experiments except for the memory and run time footprint experiment:
1. Run the bash script `run_all_experiments.sh` located in `simpa_paper_experiments/experiments`.
2. Run each python script separately in the `simpa_paper_experiments/experiments`.

Either way, the memory and run time footprint experiment has to be executed separately.
This means that the `memory_footprint.py` and `evaluate_memory_footprint_resutls.py` scripts are not meant to be executed by themselves.

## Run memory and run time footprint experiment

Change your directory to `simpa_paper_experiments/experiments/pa_image_simulation` 
Execute: `bash run_memory_footprint_experiment.sh`