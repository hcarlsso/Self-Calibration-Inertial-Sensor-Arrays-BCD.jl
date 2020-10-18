#!/bin/bash

## declare an array variable
declare -a arr=("1" "2")

## now loop through the above array
for i in "${arr[@]}"
do
    echo "$i"
    julia-1.5.2 -t ${i} -O3 time_threads.jl
    # or do whatever with individual element of the array
done
