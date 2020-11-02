#!/usr/bin/env python

'''
A python wrapper for convex (bi)-clustering binary that makes it easier to use
for matrix data clustering and the corresponding visualization.

Example: 

    ./cvxclustr.py --gamma 3,5,50,100,150,200 --col_knn 2 --row_knn 4 --tol 0.0001 --output /var/www/html/cvxclustr/res3.json --data data/president.csv
    ./cvxclustr.py --gamma 3,5,50,100,150,200,500 --col_knn 4 --row_knn 3 --tol 0.0001 --output /var/www/html/cvxclustr/lung100.json --data /var/www/html/cvxclustr/uploads/example/lung100.csv
    ./cvxclustr.py --gamma 3,5,50,100,150,200,500 --col_knn 5 --row_knn 3 --tol 0.0001 --output /var/www/html/cvxclustr/lung500.json --data /var/www/html/cvxclustr/uploads/example/lung500.csv
    python /home/haidyi/cvxclustr/cvxclustr.py --gamma 3,5,50,100,150,200 --col_knn 2 --row_knn 4 --tol 0.0001 --output /var/www/html/cvxclustr/president.json --data /var/www/html/cvxclustr/uploads/example/president.csv

Authors:  Haidong Yi <haidyi@cs.unc.edu>
          Le Huang   <lehuang@unc.edu>

Version:  2020-09-08
'''


import os
import argparse
import json
import re
import numpy as np
import scipy as sp
import pandas as pd
from scipy.io import mmwrite
from sklearn.metrics.pairwise import euclidean_distances

from tempfile import mkdtemp
from subprocess import Popen
from os.path import abspath, isfile, dirname, join as path_join
from os import devnull
from sys import stderr, stdin, stdout
from postprocessing import readMTXFile, create_tree


# define constant vars
CLUSTR_BIN_PATH = path_join(dirname(__file__), 'cvxclustr_path')
assert isfile(CLUSTR_BIN_PATH), ('unable to find the cvxclustr_path binary,'
   'please compile {} it before using this python script').format(CLUSTR_BIN_PATH)


def parse_command():
    parser = argparse.ArgumentParser('cvxclustr')

    parser.add_argument('--data', type=str, default='x.mtx', required=True, help='path to matrix file (.csv format)')
    parser.add_argument('--rgfile', type=str, default='', help='path to the row graph file')
    parser.add_argument('--cgfile', type=str, default='', help='path to the column graph file')
    parser.add_argument('--gamma', type=str, default='1', required=True, help='sequence of gamma (comma seperated)')
    parser.add_argument('--col_knn', type=int, default=2, help='k nearest neightbors used to create column weights matrix')
    parser.add_argument('--row_knn', type=int, default=4, help='k nearest neighbors used to create row weights matrix')
    parser.add_argument('--output', type=str, default='soln.json', required=True, help='the path to the output file')
    parser.add_argument('--intermediate', type=str, default='soln.json', required=False, help='the path to the output file')
    parser.add_argument('--nthreads', type=int, default=1, help='number of threads to use')
    parser.add_argument('--max_iter', type=int, default=1000, help='max iterations to run')
    parser.add_argument('--tol', type=float, default=1e-3, help='tolerance of convergence')
    parser.add_argument('--verbose', type=int, default=1, choices=[0,1], help='printing flag')
    
    args = parser.parse_args()

    # check the format of gamma
    pattern = re.compile(r"(^(\s*(\+)?\d+(?:\.\d+)?\s*,\s*)+(\+)?\d+(?:\.\d+)?\s*$)|(^(?:[1-9]\d*|0)?(?:\.\d+)?$)")
    matches = re.search(pattern, args.gamma)
    assert matches is not None, ('the input gamma {} is illegal.'.format(args.gamma))

    gamma_list = sorted([float(x) for x in args.gamma.split(',')]) # split and sort
    gamma_list = [str(x) for x in gamma_list]
    args.gamma = ','.join(gamma_list)

    return args


def _read_matrix(file):
    df = pd.read_csv(file, index_col = 0)
    data_matrix = df.to_numpy()
    
    col_names = list(df.columns)
    row_names = list(df.index)
    
    return data_matrix, col_names, row_names


def kernel_weight(X, phi=0.01):
    ''' 
    Compute kernel weights given a data matrix X and a scalar parameter.
    The weights is given by:
                  w[i, j] = exp(-phi ||x[i,:] - x[j,:]||^2)

    Parameters:
    ------------
    X: numpy.array
        The data matrix to be clustered
    phi: float 
        The nonnegative parameter that controls the scale of kernel weights
    '''
    row_wts = np.exp(-phi * euclidean_distances(X)**2)
    col_wts = np.exp(-phi * euclidean_distances(X.transpose())**2)

    return row_wts, col_wts


def knn_weight(wts, k=5):
    n, _ = wts.shape

    for i in range(n):
        sort_idx = np.argsort(wts[i,:])
        wts[i, sort_idx[:-(k+1)]] = 0
    
    np.fill_diagonal(wts, 0)
    # make weights symmetric
    wts = sp.sparse.coo_matrix( (wts + wts.transpose()) / 2)

    return wts


def write_wts_file(tmp_dir, col_wts, row_wts):
    col_wts_file = os.path.join(tmp_dir, 'col_wts.mtx')
    row_wts_file = os.path.join(tmp_dir, 'row_wts.mtx')

    mmwrite(col_wts_file, col_wts, symmetry = 'general')
    mmwrite(row_wts_file, row_wts, symmetry = 'general')

    return col_wts_file, row_wts_file


def run():
    args = parse_command()

    # preprocessing the input data
    data, col_name, row_name = _read_matrix(args.data)
    

    # create tmp dir in case of cluttering file system
    tmp_dir_path = mkdtemp()
    
    args.data = path_join(tmp_dir_path, 'data.mtx')
    mmwrite(args.data, data)
    intermiediate_outfile = path_join(tmp_dir_path, 'soln.mtx') 

    # generate wts file when no explicit input
    if args.rgfile == '' and args.cgfile == '':
        row_wts, col_wts = kernel_weight(data)
        
        row_wts = knn_weight(row_wts, args.row_knn)
        col_wts = knn_weight(col_wts, args.col_knn)
        
        args.cgfile, args.rgfile = write_wts_file(tmp_dir_path, col_wts, row_wts)

    # run the full path optimization
    with open(devnull, 'w') as dev_null:
        exec_cmd = '{} -C {} -R {} -g {} -m {} -p {} -t {} -v {} -o {} {}'.format(
            abspath(CLUSTR_BIN_PATH), args.cgfile, args.rgfile, 
            args.gamma, args.max_iter, args.nthreads, args.tol,
            args.verbose, intermiediate_outfile, args.data)
        clustr_process = Popen(
            args = exec_cmd,
            shell = True,
            stdout = stderr if args.verbose else dev_null
        )
        clustr_process.wait()
        assert not clustr_process.returncode, ('ERROR: cvxclustr_path exited '
            'with non-zero code, please enable verbose mode in command line or '
            'refer to the cvxclustr_path output for error details.')
    
    # postprocess the output and generate the result json file
    dictData = readMTXFile(intermiediate_outfile)
    
    row_dict, dictData = create_tree('row_memship', dictData)
    col_dict, dictData = create_tree('col_memship', dictData)
    
    dictionary ={  
       "rowJSON" : row_dict,  
       "colJSON" : col_dict,
       "rowName" : row_name,
       "colName" : col_name,
       "matrix0" : data.tolist()
    }

    for gamma in dictData:
       dictionary['matrix' + str(gamma)] = dictData[gamma]['heatmap'].tolist()
    with open(args.output, 'w') as out_jsfile:  
       json.dump(dictionary, out_jsfile) 
    

if __name__ == "__main__":
    run()
