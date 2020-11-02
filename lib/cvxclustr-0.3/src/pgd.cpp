#include "cvxclustr.h"



/*
 * Implement the function for setting default Options of Pgd solver
 *
 * @param opt: the ptr of pgdOpt type
 * @param arg: the ptr of program arguments
 */

void setPgdOpt(pgdOpt *opt, solver_arg *arg) {

  // set the default options for Pgd sovler
  opt->max_iters = arg->max_iter;
  opt->tol       = arg->tol;
  opt->verbose   = arg->verbose;
  opt->eta       = arg->lr;

};



/*
 * Implement an element-wise projection function
 *
 * P_{C_{d,l}}(g) = { g  if ||g||_2 < w_{d,l} \gamma_d
 *                  { w_{d,l} \gamma_d g/||g||_2 otherwise}
 *
 * @param g: input vector
 * @param w: weight parameter
 * @param gamma: weight parameter
 */
void projection_threshold(double *g, double *w_c, double *w_r, double gamma_c,
                          double gamma_r, int E_c, int E_r, int p, int n)
{
  int offset = E_c * p;
  
  #pragma omp parallel
  {
    double dnrm2, threshold;
    int j;

    #pragma omp for
    for (j = 0; j < E_c; ++j)
    {
      dnrm2 = cblas_dnrm2(p, g+j*p, 1);
      threshold = gamma_c * w_c[j] / dnrm2;
      cblas_dscal(p, threshold < 1. ? threshold : 1., g+j*p, 1);
    }

    #pragma omp for
    for (j = 0; j < E_r; ++j)
    {
      dnrm2 = cblas_dnrm2(n, g+offset+j, E_r);
      threshold = gamma_r * w_r[j] / dnrm2;
      cblas_dscal(n, threshold < 1. ? threshold : 1., g+offset+j, E_r);
    }
  }
};


double* pgd(sp_matrix *A, double *X, double *lambda0, double *w_c, double *w_r,
            double gamma_c, double gamma_r, int n, int p, int E_c, int E_r,
            pgdOpt *opt, cvx_clustr_output *out)
{
  double eta    = opt->eta;
  double tol    = opt->tol;
  int max_iters = opt->max_iters;

  const int n_row = A->m;
  const int n_col = A->n;

  double *Dlambda  = (double *) malloc(n_row * sizeof(double));
  double *lambda1  = (double *) malloc(n_row * sizeof(double));
  double *Du       = (double *) malloc(n_col * sizeof(double));
  double *Atlambda_minus_x = (double *) malloc(n_col * sizeof(double));
  double *u        = out->u_sol;
  double *_lambda1 = lambda1;

  double *temp;
  double normalizedResid;

  cblas_dcopy(n_row, lambda0, 1, lambda1, 1); // lambda1 <- lambda0
  
  printf("n_row: %d, n_col: %d\n", n_row, n_col); 
  
  spblas_dgemv(Trans, 1.0, A, lambda1, 0.0, Atlambda_minus_x);
  cblas_daxpy(n_col, -1, X, 1, Atlambda_minus_x, 1);
  out->obj[0] = 0.5 * pow(cblas_dnrm2(n_col, Atlambda_minus_x, 1), 2);

  printf("pgd itr | delta U | objFuncEval |\n");
  double start_time = gettime_();
  
  int itr;
  for (itr = 0; itr < max_iters; itr++)
  {
    temp = lambda0; lambda0 = lambda1; lambda1 = temp;

    // update lambda
    spblas_dgemv(NoTrans, eta, A, u, 0.0, lambda1); // lambda <- Au
    cblas_daxpy(n_row, 1.0, lambda0, 1, lambda1, 1);  // lambda1 <- lambda0 + eta *Au
    projection_threshold(lambda1, w_c, w_r, gamma_c, gamma_r, E_c, E_r, p, n);

    // compute delta_lambda
    cblas_dcopy(n_row, lambda0, 1, Dlambda, 1);
    cblas_daxpy(n_row, -1, lambda1, 1, Dlambda, 1);

    // update u
    spblas_dgemv(Trans, 1.0, A, Dlambda, 0.0, Du); // delta_u <- A^T delta_lambda
    cblas_daxpy(n_col, 1.0, Du, 1, u, 1);  // u <- u + Du

    normalizedResid = cblas_dnrm2(n_col, Du, 1);

    // ObjFuncEval
    spblas_dgemv(Trans, 1.0, A, lambda1, 0.0, Atlambda_minus_x);
    cblas_daxpy(n_col, -1, X, 1, Atlambda_minus_x, 1);
    out->obj[itr+1] = 0.5 * pow(cblas_dnrm2(n_col, Atlambda_minus_x, 1), 2);
    
    printf(" %5d  | %6.4f | %6.4f \n", itr+1, normalizedResid, out->obj[itr+1]);

    if (normalizedResid < tol) break;
  }

  out->itr = itr + 1; // record #itrs
  out->tau = eta;
  double end_time = gettime_();

  printf("Done: time: %.3f secs, iterations: %d\n", end_time - start_time, out->itr);

  // compute vector v
  double *v = (double *) malloc(n_row * sizeof(double));
  cblas_dcopy(n_row, lambda1, 1, v, 1);
  cblas_dscal(n_row, -1./out->tau, v, 1);
  spblas_dgemv(NoTrans, -1.0, A, out->u_sol, 1.0, v);
  
  // free memory
  free(Dlambda);
  free(_lambda1);
  free(Du);
  free(Atlambda_minus_x);

  return v;
};
