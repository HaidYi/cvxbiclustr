#include "cvxclustr.h"


/*
 * alloc memory for a sparse matrix
 * 
 * @param
 * @param     
 */
void sp_matrix_malloc(sp_matrix *mat, int m, int n, int nz, SPMAT_TYPE sptype)
{

  mat->m = m;
  mat->n = n;
  mat->nz = nz;
  mat->sptype = sptype;

  mat->data = (double *) malloc(nz * sizeof(double));
  mat->i    = (unsigned int *) malloc(nz * sizeof(unsigned int));
  
  if (sptype == SPMAT_CSC)
  {
    mat->p = (unsigned int *) malloc((n + 1) * sizeof(unsigned int));
  }
  else if (sptype == SPMAT_CSR)
  {
    mat->p = (unsigned int *) malloc((m + 1) * sizeof(unsigned int));
  }
  
}



/*
 * spblas_dgemv()
 *   Multiply a sparse matrix and a vector
 * 
 * Inputs: alpha - scalar factor
 *         A     - sparse matrix
 *         x     - dense vector
 *         beta  - scalar factor
 *         y     - (input/output) dense vector
 * 
 * Returns: y = alpha * op(A) * x + beta * y
 */
 
void spblas_dgemv (TRANSPOSE_t TransA, const double alpha, sp_matrix *A,
  double *x, const double beta, double *y)
{
    const int M = A->m;
    const int N = A->n;

    int j;
    int lenX, lenY;
    double *Ad;
    unsigned int *Ap, *Ai;

    Ap = A->p;
    Ad = A->data;

    if (TransA == NoTrans)
    {
        lenX = N;
        lenY = M;
    }
    else
    {
        lenX = M;
        lenY = N;
    }

    if (beta == 0.0)
    {
        for (j = 0; j < lenY; ++j)
        {
            y[j] = 0.0;
        }
    }
    else if (beta != 1.0)
    {
        cblas_dscal(lenY, beta, y, 1);
    }
    
    if (alpha == 0.0)
    {
        return;
    }

    if ((SPMATRIX_ISCSC(A) && TransA == NoTrans) ||
        (SPMATRIX_ISCSR(A) && TransA == Trans  ))
    {
        Ai = A->i;
        
        for (j = 0; j < lenX; ++j)
        {
            unsigned int p;
            for (p = Ap[j]; p < Ap[j + 1]; ++p)
            {
                y[Ai[p]] += alpha * Ad[p] * x[j];
            }
        }
    }
    else if ((SPMATRIX_ISCSC(A) && TransA == Trans) ||
             (SPMATRIX_ISCSR(A) && TransA == NoTrans))
    {
        Ai = A->i;

        #pragma omp parallel for
        for (j = 0; j < lenY; ++j)
        {
            unsigned int p;
            for (p = Ap[j]; p < Ap[j + 1]; ++p)
            {
                y[j] += alpha * Ad[p] * x[Ai[p]];
            }
        }
    }
    else
    {
        fprintf(stderr, "invalid argument: %s\n", strerror(EINVAL));
    }
    
};


void spblas_dgemv_blocked(TRANSPOSE_t TransA, const double alpha, sp_matrix *A, double *x,
                          const double beta, double *y, int n_select, int *idx, int block_size)
{
  const int M = A->m;
  const int N = A->n;

  int j;
  int lenX, lenY;
  double *Ad;
  unsigned int *Ap, *Ai;

  Ap = A->p;
  Ad = A->data;

  if (TransA == NoTrans)
  {
    lenX = N;
    lenY = M;
  }
  else
  {
    lenX = M;
    lenY = N;
  }

  if (beta == 0)
  {
    for (j = 0; j < lenY; ++j)
    {
      y[j] = 0.0;
    }
  }
  else if (beta != 1.0)
  {
    cblas_dscal(lenY, beta, y, 1);
  }

  if (alpha == 0.0)
  {
    return;
  }

  if ((SPMATRIX_ISCSC(A) && TransA == NoTrans) ||
      (SPMATRIX_ISCSR(A) && TransA == Trans ))
  {
    Ai = A->i;
    // to do ...
#pragma omp parallel for
    for(j = 0; j < n_select; ++j)
    {
      int k;
      int offset = block_size * idx[j];

      for (k = 0; k < block_size; ++k)
      {
        unsigned int p;
        for (p = Ap[offset+k]; p < Ap[offset+k+1]; ++p)
        {
          #pragma omp atomic update
          y[Ai[p]] += alpha * Ad[p] * x[offset+k];
        }
      }
    }
  } 
  else if ((SPMATRIX_ISCSC(A) && TransA == Trans) ||
           (SPMATRIX_ISCSR(A) && TransA == NoTrans))
  { 
    Ai = A->i;
    // to do ...

#pragma omp parallel for
    for (j = 0; j < n_select; ++j)
    {
      int k;
      int offset = block_size * idx[j];

      for (k = 0; k < block_size; ++k)
      {
        unsigned int p;
        for (p = Ap[offset+k]; p < Ap[offset+k+1]; ++p)
        {
          y[offset+k] += alpha * Ad[p] * x[Ai[p]];
        }
      }
    }
    
  } 
  else
  { 
    fprintf(stderr, "invalid argument: %s\n", strerror(EINVAL));
  } 
    
};  
    

void sp_matrix_copy(sp_matrix *src, sp_matrix *dest)
{
  if ((src->m != dest->m) || (src->n != dest->n))
  {
    ERROR_MSG("the size of src and dest matrix is inconsistent")
  }
  else if (src->nz != dest->nz)
  {
    ERROR_MSG("the number of non-zero elements in src and dest is inconsistent");
  }
  else if (src->sptype != dest->sptype)
  {
    ERROR_MSG("the sparse type of src and dest matrix is inconsistent")
  }
  else
  {
    int j;
    for (j = 0; j < dest->nz; ++j)
    {
      dest->data[j] = src->data[j];
      dest->i[j] = src->i[j];
    }

    if (dest->sptype == SPMAT_CSR)
    {
      for (j = 0; j < dest->m+1; ++j)
      {
        dest->p[j] = src->p[j];
      }
    }
    else
    {
      for (j = 0; j < dest->n+1; ++j)
      {
        dest->p[j] = src->p[j];
      }
    }
    
  }

}



void sp_matrix_csr_tocsc(sp_matrix *src, sp_matrix *dest)
{
  if (src->sptype == SPMAT_CSR && dest->sptype == SPMAT_CSC)
  {
    const int nnz = dest->nz;
    int j;
    
    for (j = 0; j <= dest->n; ++j)
    {
      dest->p[j] = 0;
    }
    
    for (j = 0; j < nnz; ++j)
    {
      dest->p[src->i[j]]++;  
    }

    int col, cumsum;
    for (col = 0, cumsum = 0; col < dest->n; ++col)
    {
      int temp = dest->p[col];
      dest->p[col] = cumsum;
      cumsum += temp;
    }
    dest->p[dest->n] = nnz;

    int row, last;
    for (row = 0; row < dest->m; ++row)
    {
      int jj;
      for (jj = src->p[row]; jj < src->p[row+1]; ++jj)
      {
        int col = src->i[jj];
        int dst = dest->p[col];

        dest->i[dst] = row;
        dest->data[dst] = src->data[jj];

        dest->p[col]++;
      }

    }

    for (col = 0, last = 0; col <= dest->n; ++col)
    {
      int temp     = dest->p[col];
      dest->p[col] = last;
      last         = temp;
    }
  }
  else
  {
    ERROR_MSG("either the type of src or dest is inconsistent")
  }
  

}



void sp_matrix_csc_tocsr(sp_matrix *src, sp_matrix *dest)
{
  if (src->sptype == SPMAT_CSC && dest->sptype == SPMAT_CSR)
  {
    sp_matrix srcT;
    sp_matrix_transpose(src, &srcT);

    sp_matrix_transpose(dest);  // csc
    sp_matrix_csr_tocsc(&srcT, dest);  // csr to csc
    sp_matrix_transpose(dest);  // csr
  }
  else
  {
    ERROR_MSG("either the type of src or dest is inconsistent");
  }
  
}


void sp_matrix_transpose(sp_matrix *spm, sp_matrix *spmT)
{
  if (spm->sptype == SPMAT_CSR)
  {
    spmT->sptype = SPMAT_CSC;
  }
  else if (spm->sptype == SPMAT_CSC)
  {
    spmT->sptype = SPMAT_CSR;
  }

  spmT->m = spm->n;
  spmT->n = spm->m;

  spmT->i = spm->i;
  spmT->p = spm->p;
  spmT->data = spm->data;
  
}


/*
 * inplace sparse matrix transpose
 * 
 * @param  m: csr or csc sparse matrix
 * @return 
 */

void sp_matrix_transpose(sp_matrix *spm)
{   
    if (spm->sptype == SPMAT_CSR)
    {
        spm->sptype = SPMAT_CSC;
    }
    else if (spm->sptype == SPMAT_CSC)
    {
        spm->sptype = SPMAT_CSR;
    }

    unsigned int temp;
    temp = spm->m;
    spm->m = spm->n;
    spm->n = temp;
    
};


/* 
 * free the memory allocated for csc matrix
 * 
 * 
 * @param m: the ptr to csc_matrix
 */
void sp_matrix_free(sp_matrix *m)
{
    free(m->p);
    free(m->i);
    free(m->data);
};



/*
 * display sparse matrix with csr or csc format
 * 
 * 
 * @param spm: pointer to sparse matrix
 */
void sp_matrix_fprintf(FILE *stream, sp_matrix *A)
{
    char msg[MAX_STR_LENGTH];
    const char *s = A->sptype == SPMAT_CSR ? "row" : "column";
    sprintf(msg, "matrix in compressed %s format:\n", s);
    fprintf(stream, "%s", msg);

    unsigned int i;
    fprintf(stream, "i = [ ");
    for (i = 0; i < A->nz; ++i)
    {
        fprintf(stream, "%d, ", A->i[i]);
    }
    fprintf(stream, "]\n");

    fprintf(stream, "p = [ ");
    unsigned int size = A->sptype == SPMAT_CSR ? A->m : A->n;
    for (i = 0; i < size + 1; ++i)
    {
        fprintf(stream, "%d, ", A->p[i]);
    }
    fprintf(stream, "]\n");

    fprintf(stream, "d = [ ");
    for (i = 0; i < A->nz; ++i)
    {
        fprintf(stream, "%g, ", A->data[i]);
    }
    fprintf(stream, "]\n");

};
