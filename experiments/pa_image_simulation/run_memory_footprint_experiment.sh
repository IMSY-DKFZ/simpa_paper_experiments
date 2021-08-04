#!/bin/sh

current_dir=$(pwd)
tmp_dir="$current_dir/tmp_memory_footprint"
mkdir -p $tmp_dir
if [ $1 = "c" ]
then
  :
else
  rm $tmp_dir/*
fi

spacing_list="0.4 0.35 0.2 0.15"
repeat_per_spacing=1


for spacing in $spacing_list
do
  for (( i=0; i < $repeat_per_spacing; ++i ))
  do
    mprof run --output "$tmp_dir/mem_spacing_$spacing-run_$i.csv" --interval 0.01 memory_footprint.py --spacing $spacing
  done
done

python3 evaluate_memory_footprint_results.py --spacing_list "$spacing_list" --output_dir "$tmp_dir"