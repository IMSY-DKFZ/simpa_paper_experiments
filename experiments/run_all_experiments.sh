#!/bin/sh

current_dir=$(pwd)

for file in "$current_dir/full_simulation"/*
do
  python3 "$file"
done

for file in "$current_dir/tissue_generation"/*
do
  python3 "$file"
done

for file in "$current_dir/independent_execution"/*
do
  python3 "$file"
done

for file in "$current_dir/pa_image_processing"/*
do
  python3 "$file"
done

python3 "$current_dir/pa_image_simulation/realistic_forearm_simulations.py"
python3 "$current_dir/pa_image_simulation/pipeline_hyperparam_configurations.py"
python3 "$current_dir/pa_image_simulation/example_of_modularity.py"
python3 "$current_dir/pa_image_simulation/generate_forearm_dataset.py"
cd pa_image_simulation
bash run_memory_footprint_experiment.sh
