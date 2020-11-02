#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.69])
AC_INIT([cvxclustr], [0.3], [haidyi@cs.unc.edu])


# Checks for programs.
AC_PROG_CXX
AC_PROG_CC
AM_INIT_AUTOMAKE
AC_PROG_LIBTOOL

# Checks for command line options
AC_ARG_ENABLE([omp-exec],
    [AS_HELP_STRING([--disable-omp-exec],
        [disable openmp execution @<:@default: no@:>@])],
    [omp_exec=${enableval}], [omp_exec=yes])

if test "x${omp_exec}" = "xyes"; then
    AC_OPENMP

    if test "x$OPENMP_CFLAGS" = "x"; then
        AC_MSG_WARN([
  ---------------------------------------
   Unable to find openmp on this system.
   Building a single-threaded version.
  ---------------------------------------])
        omp_exec=no
    else
        AC_SUBST(AM_CFLAGS,"$OPENMP_CFLAGS")
    fi
fi

# Checks for libraries.
AC_CHECK_LIB([m], [pow])
AC_CHECK_LIB([igraph],
    [igraph_empty],
    [], [AC_MSG_ERROR([
  ---------------------------------------
   Unable to find library igraph around.
   Install the dependency igraph first.
  ---------------------------------------])])
AC_CHECK_LIB([openblas],
    [cblas_daxpy],
    [], [AC_MSG_ERROR([
  ---------------------------------------
   Unable to find library openblas around.
   Install the dependency openblas first.
  ---------------------------------------])])

# Checks for header files.
AC_CHECK_HEADERS([float.h stdint.h sys/time.h])

# Checks for typedefs, structures, and compiler characteristics.
AC_CHECK_HEADER_STDBOOL
AC_TYPE_SIZE_T

# Checks for library functions.
AC_FUNC_MALLOC
AC_CHECK_FUNCS([floor gettimeofday pow sqrt strerror])

AC_CONFIG_FILES([Makefile
                 cvxclustr.pc
                 src/Makefile])
AC_OUTPUT
