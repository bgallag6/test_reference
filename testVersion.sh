#!/bin/bash

echo "The process of this program:
1) Creates datacube of raw images (solar images are derotated)
2) Extracts timeseries' of pixel intensity values with corresponding timestamps and exposure durations
3) Power-spectra are computed from extracted timeseries using the Fast Fourier Transform
4) Models are fit to the spectra and their parameters extracted"


while getopts ":n:" opt; do
  case $opt in
    n) num=$OPTARG;;
    \?) echo "Invalid option: -$OPTARG" >&2; exit 1;;
    :) echo "Option -$OPTARG requires an argument." >&2; exit 1;;
  esac
done

if [ $num -gt 1 ]; then
    tey="python"
else
    tey=
fi 

$tey python sysTest1.py