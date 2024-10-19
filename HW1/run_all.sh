#!/bin/bash

# Run all executables and redirect output to CSV files
./mv_mult_contiguous > results/results_contiguous.csv
./mv_mult_separate_rows > results/results_separate_rows.csv
./mv_mult_loop_unrolling > results/results_loop_unrolling.csv
./mv_mult_padding > results/results_padding.csv
./mv_mult_column_major > results/results_column_major.csv

echo "All executables have been run, and results are saved to CSV files."
