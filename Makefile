#!/bin/bash

CC=g++
CFLAGS=-O3 -std=c++11 -fopenmp
LDFLAGS=-lcvxclustr -ligraph -lopenblas

all: cvxclustr_path

cvxclustr_path: cvxclustr_path.cpp mmio.cpp
	$(CC) $(CFLAGS) cvxclustr_path.cpp mmio.cpp -o cvxclustr_path $(LDFLAGS)
	
clean:
	rm -vf main main.o
