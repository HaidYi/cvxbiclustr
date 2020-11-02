lib_LTLIBRARIES = libcvxclustr.la
libcvxclustr_includedir= ${includedir}/cvxclustr
libcvxclustr_include_HEADERS=cvxclustr.h
libcvxclustr_la_CXXFLAGS=-std=c++11 -fopenmp
libcvxclustr_la_SOURCES = cvxclustr.cpp fasta.cpp pgd.cpp spblas.cpp util.cpp

pkgconfigdir=$(libdir)/pkgconfig
pkgconfig_DATA= ../cvxclustr.pc
