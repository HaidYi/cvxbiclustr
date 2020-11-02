/*
 * The header file for Convex (Bi)clustering
 * 
 * 
*/

#ifndef CVXCLUSTR_H
#define CVXCLUSTR_H

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdbool.h>
#include <sys/time.h>
#include <math.h>
#include <float.h>
#include <stdint.h>
#include <errno.h>
#include <omp.h>
#include <cblas.h>
#include <igraph/igraph.h>

// Constant Var
#define MAX_STR_LENGTH 1024
#define PATH_JOIN_SEP "/"
#define EPS 1e-12

typedef enum {ROW, COLUMN} CVX_DIM;
typedef enum {Trans, NoTrans} TRANSPOSE_t;
typedef enum {SPMAT_CSR, SPMAT_CSC} SPMAT_TYPE;

// Exception

#define ERR_FILE __FILE__
#define ERR_LINE __LINE__

#define ERROR_MSG(msg)                                      \
  { fprintf(stderr, "\033[31m%s, line %d: %s\033[0m\n",     \
      ERR_FILE, ERR_LINE, (msg));                           \
    abort();                                                \
  }
#define ERROR              ERROR_MSG ("")
#define NOT_IMPLEMENTED    ERROR_MSG ("Not implemented: ")
#define NEVER_CALL         ERROR_MSG ("Never call: ")
#define NOT_FOUND          ERROR_MSG ("Not found: ")

// Log 

#define ERROR_INFO(msg)                                   \
  {                                                       \
    fprintf(stderr, "\033[31m%s\033[0m\n", (msg)); \
  }
#define WARN_INFO(msg)                                    \
  {                                                       \
    printf("\033[33m%s \033[0m\n", (msg));       \
  }
#define LOG_INFO(msg)       \
  {                         \
    printf("%s\n", (msg));  \
  }

// define func

#define SPMATRIX_ISCSR(src) (src->sptype==SPMAT_CSR)
#define SPMATRIX_ISCSC(src) (src->sptype==SPMAT_CSC)

#define min(a,b) ((a) < (b) ? (a) : (b))
#define max(a,b) ((a) > (b) ? (a) : (b))

// Type declaration

typedef struct {
    bool verbose;
    int max_iter;
    double tol, lr;
} solver_arg;


typedef struct {
    int m, n;
    unsigned int *p;
    unsigned int *i;
    double *data;
    size_t sptype, nz;
} sp_matrix;


typedef struct
{
    unsigned int _from, _to;
} edge;


typedef struct
{
    int n, p, E_c, E_r;
    double gamma_c, gamma_r;
    sp_matrix *A;
    edge *e_c, *e_r;
    double *x, *w_c, *w_r;
    int *D;
} cvx_clustr_param;

// Options needed by Fasta solver
typedef struct
{
    unsigned int max_iters;
    double tol;
    bool verbose;
    bool recordObjective;
    bool adaptive;
    bool backtrack;
    unsigned int seed; 
    double stepsizeShrink;
    int window;
    double eps_r;
    double eps_n;
    double L;
    double tau;
} fastaOpt;

// Options needed by Project Gradient solver
typedef struct
{
  unsigned int max_iters;
  double tol;
  bool verbose;
  double eta;  // learning rate
} pgdOpt;


typedef struct
{
    unsigned int itr;
    double tau;
    double *obj;
    double *u_sol;
    double *v_sol;

    igraph_vector_t row_memship, col_memship;
    igraph_vector_t csize_row, csize_col;
    int no_row, no_col;

} cvx_clustr_output;


// cvxclustr.c
void cvxclustr(cvx_clustr_param *param, cvx_clustr_output *out, char *solver,
    int max_iter, double tol, double lr, bool verbose);

// util.c
double gettime_(void);
char* path_join(const char *, const char*);
void param_free(cvx_clustr_param *param);
void darray_set_all(double *v, double val, int n);
double random_gaussian(const double sigma);
int* prox(double *v, double gamma_c, double gamma_r, double *w_c, double *w_r, double tau,
  int E_c, int E_r, int n, int p);
void create_graph(int *ind, igraph_t *r_g, igraph_t *c_g, edge *e_r, edge *e_c, int E_r, int E_c);
void double_vector_fprintf(FILE *stream, double *v, int n);

// fasta.c
void fasta_gradf(double *x, double *Alambda, double *y, int *D, int n);
double fasta_feval(double *x, double *Alambda, int *D, int n);
void setFastaOpt(fastaOpt *opt, sp_matrix *A, double *X, int *D, solver_arg *arg);
double* fasta(sp_matrix *A, double *X, double *x0, double *w_c, double *w_r, int *D, double gamma_c, double gamma_r, 
  int n, int p, int E_c, int E_r, fastaOpt *opt, cvx_clustr_output *out);

// pgd.c
void setPgdOpt(pgdOpt *opt, solver_arg *args);
double* pgd(sp_matrix *A, double *X, double *lambda0, double *w_c, double *w_r,
            double gamma_c, double gamma_r, int n, int p, int E_c, int E_r,
            pgdOpt *opt, cvx_clustr_output *out);

// spblas.c
void sp_matrix_malloc(sp_matrix *mat, int m, int n, int nz, SPMAT_TYPE sptype);
void spblas_dgemv (TRANSPOSE_t TransA, const double alpha, sp_matrix *A, double *x, const double beta, double *y);
void sp_matrix_free(sp_matrix *m);
void sp_matrix_fprintf(FILE *stream, sp_matrix *A);
void sp_matrix_transpose(sp_matrix *spm);

void sp_matrix_copy(sp_matrix *src, sp_matrix *dest);
void sp_matrix_csr_tocsc(sp_matrix *src, sp_matrix *dest);
void sp_matrix_csc_tocsr(sp_matrix *src, sp_matrix *dest);
void sp_matrix_transpose(sp_matrix *spm, sp_matrix *spmT);


#endif
