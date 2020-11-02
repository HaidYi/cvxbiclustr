#!/bin/bash

BIN_PROGRAM=/nas/longleaf/home/haidyi/proj/cvxclustr/cvxclustr.py
data_path=/nas/longleaf/home/haidyi/proj/cvxclustr/data
nrun=5

# w/ compression
for i in $(seq 1 $nrun)
do
  echo "compressed: $i"

  ${BIN_PROGRAM} \
    --gamma 1,10,100,1000,5000,10000 \
    --col_knn 5 \
    --row_knn 3 \
    --tol 0.001 \
    --max_iter 10000 \
    --output ${data_path}/result.json \
    --data ${data_path}/dataset/tcga_breast.csv \
    --verbose 1
done

# w/o compression
gamma_list=("1" "10" "100" "1000" "5000" "10000")
  
for i in $(seq 1 $nrun)
do
  echo "non-compressed: $i"
  for gamma in ${gamma_list[@]}
    do
      ${BIN_PROGRAM} \
      --gamma ${gamma} \
      --col_knn 5 \
      --row_knn 3 \
      --tol 0.001 \
      --max_iter 10000 \
      --output ${data_path}/result.json \
      --data ${data_path}/dataset/tcga_breast.csv \
      --verbose 1
  done
done

exit
