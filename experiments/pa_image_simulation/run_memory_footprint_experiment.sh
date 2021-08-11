#!/bin/sh

current_dir=$(pwd)
tmp_dir="$current_dir/tmp_memory_footprint"
mkdir -p $tmp_dir

min_spacing=0.39
max_spacing=0.4
step_spacing=0.01
spacing_list=$(python3 -c "import numpy as np; print(np.arange($min_spacing, $max_spacing + $step_spacing, $step_spacing))")
spacing_list=${spacing_list:1:-1}

repeat_per_spacing=1


for spacing in $spacing_list
do
  for (( i=0; i < $repeat_per_spacing; ++i ))
  do
    if [ $1 = "c" ]
    then
      :
    else
      rm "$tmp_dir/mem_spacing_$spacing-run_$i.csv"
    fi
    mprof run --output "$tmp_dir/mem_spacing_$spacing-run_$i.csv" --interval 0.01 memory_footprint.py --spacing $spacing
  done
done

python3 evaluate_memory_footprint_results.py --spacing_list "$spacing_list" --output_dir "$tmp_dir"