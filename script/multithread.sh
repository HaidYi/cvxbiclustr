#!/bin/bash

BIN_PROGRAM=/nas/longleaf/home/haidyi/proj/cvxclustr/cvxclustr.py
data_path=/nas/longleaf/home/haidyi/proj/cvxclustr/data
nrun=5

# run with different threads
nthread_list=("1" "4" "8" "16")
  

for nthread in ${nthread_list[@]}
do
  echo "nthread: $nthread"
  
  for i in $(seq 1 $nrun)
  do
      echo "nrun: $i"
      ${BIN_PROGRAM} \
      --gamma 100 \
      --col_knn 20 \
      --row_knn 20 \
      --tol 0.001 \
      --max_iter 10000 \
      --nthread $nthread \
      --output ${data_path}/result.json \
      --data ${data_path}/dataset/tcga_breast.csv \
      --verbose 1
  done
done

exit
