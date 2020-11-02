#include "cvxclustr.h"


void fasta_gradf(double *x, double *Alambda, double *y, int *D, int n)
{
    int j;
    
    #pragma omp parallel for
    for (j = 0; j < n; ++j)
    {
        y[j] = (1./D[j]) * Alambda[j] - x[j];
    }
};


double fasta_feval(double *x, double *Alambda, int *D, int n)
{
    double nrm22 = 0;
    int j;
    
    for (j = 0; j < n; ++j)
    {
        nrm22 += D[j] * pow(x[j] - (1./D[j]) * Alambda[j], 2);
    }

    return 0.5 * nrm22;
};


void proxg(double *x, double* res, double *w_c, double *w_r, double gamma_c, double gamma_r, int p, int n, int E_c, int E_r)
{
    int offset = E_c * p;
    
    #pragma omp parallel
    {
        double dnrm2, thres;
        int j;
        
        #pragma omp for
        for (j = 0; j < E_c; ++j)
        {
            cblas_dcopy(p, x+j*p, 1, res+j*p, 1);
            dnrm2 = cblas_dnrm2(p,   res+j*p, 1);
            thres = gamma_c * w_c[j] / dnrm2;
            cblas_dscal(p, thres < 1 ? thres : 1, res+j*p, 1);
        }
        
        #pragma omp for
        for (j = 0; j < E_r; ++j)
        {
            cblas_dcopy(n, x+offset+j, E_r, res+offset+j, E_r);
            dnrm2 = cblas_dnrm2(n, x+offset+j, E_r);
            thres = gamma_r * w_r[j] / dnrm2;
            cblas_dscal(n, thres < 1 ? thres : 1, res+offset+j, E_r);
        }
    
    }
   
};


double max_subarr(double *p, int start, int end)
{
    int j;
    double _maxValue = p[start];

    for (j = start; j <= end; ++j)
    {
        if (p[j] > _maxValue)
        {
            _maxValue = p[j];
        }
    }

    return _maxValue;
};


void setFastaOpt(fastaOpt *opt, sp_matrix *A, double *X, int *D, solver_arg *arg)
{
    opt->max_iters       = arg->max_iter;
    opt->tol             = arg->tol;
    opt->verbose         = arg->verbose;
    opt->recordObjective = true;
    opt->adaptive        = true;
    opt->backtrack       = true;
    opt->stepsizeShrink  = 0.2;
    opt->seed            = 111;

    opt->window =   10;
    opt->eps_r  = 1e-8;
    opt->eps_n  = 1e-8;
    
    // set opt->L and opt->tau

    // set random seed
    srand(opt->seed);

    double *x1     = (double *) malloc(sizeof(double) * A->n);
    double *x2     = (double *) malloc(sizeof(double) * A->n);
    double *tmp    = (double *) malloc(sizeof(double) * A->m);
    double *gradf1 = (double *) malloc(sizeof(double) * A->n);
    double *gradf2 = (double *) malloc(sizeof(double) * A->n);

    int j;
    for (j = 0; j < A->n; ++j)
    {
        x1[j] = random_gaussian(1.0);
        x2[j] = random_gaussian(1.0);
    }

    spblas_dgemv(NoTrans, 1.0, A, x1, 0.0, tmp);
    fasta_gradf(X, tmp, tmp, D, A->m);
    spblas_dgemv(Trans, 1.0, A, tmp, 0.0, gradf1);

    spblas_dgemv(NoTrans, 1.0, A, x2, 0.0, tmp);
    fasta_gradf(X, tmp, tmp, D, A->m);
    spblas_dgemv(Trans, 1.0, A, tmp, 0.0, gradf2);

    cblas_daxpy(A->n, -1, gradf1, 1, gradf2, 1); // gradf2 = gradf2 - gradf1
    cblas_daxpy(A->n, -1, x1,     1, x2,     1); // x2 = x2 - x1

    opt->L = cblas_dnrm2(A->n, gradf2, 1) / cblas_dnrm2(A->n, x2, 1);
    opt->L = opt->L > 1e-6 ? opt->L : 1e-6; 
    opt->tau = 2 / (opt->L * 10);

    free(x1);
    free(x2);
    free(tmp);
    free(gradf1);
    free(gradf2);

};


double* fasta(sp_matrix *A, double *X, double *x0, double *w_c, double *w_r, int *D, double gamma_c, double gamma_r, 
  int n, int p, int E_c, int E_r, fastaOpt *opt, cvx_clustr_output *out)
{
    double tau1   = opt->tau;
    int max_iters = opt->max_iters;
    int W         = opt->window;

    // allocate memory
    double *residual   = (double*) calloc(max_iters,   sizeof(double));
    double *normalizedResid = (double*) calloc(max_iters, sizeof(double));
    double *taus       = (double*) calloc(max_iters,   sizeof(double));
    double *fVals      = (double*) calloc(max_iters,   sizeof(double));

    // trick: copy A using CSR format for parallel computing
    sp_matrix A_csr;
    sp_matrix_malloc(&A_csr, A->m, A->n, A->nz, SPMAT_CSR);
    sp_matrix_csc_tocsr(A, &A_csr);
    
    int totalBacktracks = 0;
    int backtrackCount  = 0;
    double newObjectiveValue;
    // double bestObjectiveValue;

    // Initialize array values
    const int n_row = A->m;   // nrow
    const int n_col = A->n;   // ncol

    double *x1hat   = (double*) malloc(n_col * sizeof(double));
    double *x1      = (double*) malloc(n_col * sizeof(double));
    double *gradf0  = (double*) malloc(n_col * sizeof(double));
    double *gradf1  = (double*) malloc(n_col * sizeof(double));
    double *d1      = (double*) malloc(n_row * sizeof(double));
    double *Dx      = (double*) malloc(n_col * sizeof(double));
    double *Dg      = (double*) malloc(n_col * sizeof(double));
    double *tmp_m   = (double*) malloc(n_row * sizeof(double));
    
    double tau0, f1;
    double *temp;
    double *_x1 = x1;

    cblas_dcopy(n_col, x0, 1, x1, 1);  // x1 <- x0
    spblas_dgemv(NoTrans, 1.0, &A_csr, x1, 0.0, d1); // d1 = A(x1)

    f1 = fasta_feval(X, d1, D, n_row);                // f1 = f(d1)
    fVals[0] = f1;                           // fVals(1) = f1;
    fasta_gradf(X, d1, tmp_m, D, n_row);
    spblas_dgemv(Trans, 1.0, A, tmp_m, 0.0, gradf1);  // gradf1 = At(gradf(d1));

    // handle non-monotonicity
    double maxResidual       = DBL_MIN;
    double minObjectiveValue = DBL_MAX;

    // If user has chosen to record objective, then record initial value
    if (opt->recordObjective)
    {
        out->obj[0] = f1;
    }

    double start_time = gettime_();

    int itr, j;
    for (itr = 0; itr < max_iters; ++itr)
    {
        temp = x0; x0 = x1; x1 = temp;                  // x0 = x1;
        temp = gradf0; gradf0 = gradf1; gradf1 = temp;  // grad0 = grad1;
        tau0 = tau1;                                    // tau0  = tau1;

        // FBS step
        for (j = 0; j < n_col; j++)
        {
            x1hat[j] = x0[j] - tau0 * gradf0[j];
        }

        proxg(x1hat, x1, w_c, w_r, gamma_c, gamma_r, p, n, E_c, E_r);
        

        for (j = 0; j < n_col; j++)
        {
            Dx[j] = x1[j] - x0[j];
        }

        spblas_dgemv(NoTrans, 1.0, &A_csr, x1, 0.0, d1);
        f1 = fasta_feval(X, d1, D, n_row);

        if (opt->backtrack)
        {
            double M = max_subarr(fVals, itr - W > 0 ? itr - W: 0, itr - 1 > 0 ? itr - 1: 0);
            backtrackCount = 0;
            while (f1 - 1e-12 > M + cblas_ddot(n_col, Dx, 1, gradf0, 1) + cblas_ddot(n_col, Dx, 1, Dx, 1) / (2 * tau0) && backtrackCount < 20)
            {
                tau0 = tau0 * opt->stepsizeShrink;
                for (j = 0; j < n_col; j++)
                {
                    x1hat[j] = x0[j] - tau0 * gradf0[j];
                }

                proxg(x1hat, x1, w_c, w_r, gamma_c, gamma_r, p, n, E_c, E_r);
                
                spblas_dgemv(NoTrans, 1.0, &A_csr, x1, 0.0, d1);
                f1 = fasta_feval(X, d1, D, n_row);
                for (j = 0; j < n_col; j++)
                {
                    Dx[j] = x1[j] - x0[j];
                }
                backtrackCount += 1;
            }
            totalBacktracks += backtrackCount;
            
        }

        taus[itr] = tau0;
        residual[itr] = cblas_dnrm2(n_col, Dx, 1) / tau0;
        maxResidual = maxResidual > residual[itr] ? maxResidual : residual[itr];

        double normalizer = 0;
        for (j = 0; j < n_col; j++)
        {
            normalizer += pow(x1[j] - x1hat[j], 2);
        }
        normalizer = max(cblas_dnrm2(n_col, gradf0, 1), pow(normalizer, 0.5) / tau0) + opt->eps_n;
        normalizedResid[itr] = residual[itr] / normalizer;
        fVals[itr] = f1;

        if (opt->recordObjective)
        {
            out->obj[itr+1] = f1;
            newObjectiveValue = out->obj[itr+1];
        }
        else
        {
            newObjectiveValue = residual[itr];  // Use the residual to evaluate quality of iterate if we don't have objective
        }

        if (newObjectiveValue < minObjectiveValue)
        {
            minObjectiveValue = newObjectiveValue;
        }

        if (opt->verbose)
        {
            // fprintf(stdout, "%d: resid = %0.2f, backtrack = %d, tau = %f, objective = %0.4f\n", itr+1, residual[itr], backtrackCount, tau0, out->obj[itr+1]);
        }

        // check stopping criteria
        if (residual[itr] / (maxResidual + opt->eps_r) < opt->tol || normalizedResid[itr] < opt->tol || itr >= max_iters-1)
        {
            // collect results
            out->itr = itr + 1;
            out->tau = tau1;
            
            break;
        }

        // compute next stepsize
        if (opt->adaptive)
        {
            fasta_gradf(X, d1, tmp_m, D, n_row);
            spblas_dgemv(Trans, 1.0, A, tmp_m, 0.0, gradf1);  // gradf1 = At(gradf(d1));

            for (j = 0; j < n_col; ++j)
            {
                Dg[j] = gradf1[j] + (x1hat[j] - x0[j]) / tau0;
            }
            double dotprod = cblas_ddot(n_col, Dx, 1, Dg, 1);
            double tau_s = cblas_ddot(n_col, Dx, 1, Dx, 1) / dotprod;
            double tau_m = dotprod / cblas_ddot(n_col, Dg, 1, Dg, 1);
            tau_m = tau_m > 0 ? tau_m : 0;
            if (2 * tau_m > tau_s)
            {
                tau1 = tau_m;
            }
            else
            {
                tau1 = tau_s - .5 * tau_m;
            }
            if (tau1 <= 0 || isinf(tau1) || isnan(tau1))
            {
                tau1 = tau0 * 1.5;
            }
        }

    }

    double end_time = gettime_();
    fprintf(stdout, "\tDone: time = %0.3f secs, iterations = %d\n", end_time - start_time, out->itr);

    // compute the optimal matrix u
    spblas_dgemv(NoTrans, -1, &A_csr, x1, 0.0, d1);
    for (j = 0; j < n_row; ++j)
    {
        out->u_sol[j] += d1[j] / D[j];
    }

    // compute vector v
    double *v = (double *) malloc(n_col * sizeof(double));
    cblas_dcopy(n_col, x1, 1, v, 1);
    cblas_dscal(n_col, -1./out->tau, v, 1);
    spblas_dgemv(Trans, -1.0, A, out->u_sol, 1.0, v);

    // free memory
    sp_matrix_free(&A_csr);

    free(x1hat);
    free(_x1);
    free(d1);
    free(Dx);
    free(Dg);
    free(gradf0);
    free(gradf1);
    free(tmp_m);

    free(residual);
    free(normalizedResid);
    free(taus);
    free(fVals);

    return v;

}
