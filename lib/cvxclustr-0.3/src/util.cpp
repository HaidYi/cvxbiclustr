#include "cvxclustr.h"


/*
 * utility function for getting time
 *
 * @return current time
*/ 
double gettime_(void) {
    struct timeval timer;
    if (gettimeofday(&timer, NULL))
        return -1.0;
    return timer.tv_sec + 1.0e-6 * timer.tv_usec;
}




/*
 * ulitity function for save weights
 * 
 * 
 * @param dest : double array used for hold weights
 * @param src  : source struct array 
 */
// void save_wts(double *dest, struct edge *src, int n)
// {
//     for (int i = 0; i < n; i++)
//     {
//         dest[i] = src[i].wts;
//     }
// }


/*
 * ulitity function for load weights
 * 
 * 
 * @param : target struct edge array to load weights
 * @param : source double array
 */
// void load_wts(struct edge *dest, double *src, int n)
// {
//     for (int i = 0; i < n; i++)
//     {
//         dest[i].wts = src[i];
//     }
// }


/*
 * utility function for writing vec result
 * 
 * @param v   : the pointer of a gsl vector
 * @param dir : the path you want to write
 */
// void write_vec(gsl_vector *v, char *dir)
// {
//     FILE *fp;
//     if ( (fp = fopen(dir, "w")) != NULL )
//     {
//         printf("writing solution to %s.\n", dir);
//         gsl_vector_fprintf(fp, v, "%lf");
//         fclose(fp);
//     }
//     else
//     {
//         printf("\033[31mERROR: either you don't type the output path or the path you type doesn't exist.\033[0m\n");
//     }
    
// }


/*
 * utility function for wrting mat result
 * 
 * 
 * 
 * @param m   : the pointer of a gsl matrix
 * @param dir : the path you want to write
 */
// void write_mat(gsl_matrix *m, char *dir)
// {
//     FILE *fp;
//     if ( (fp = fopen(dir, "w")) != NULL )
//     {
//         printf("writing solution to %s.\n", dir);
//         gsl_matrix_fprintf(fp, m, "%lf");
//         fclose(fp);
//     }
//     else
//     {
//         printf("\033[31mERROR: either you don't type the output path or the path you type doesn't exist.\033[0m\n");
//     }
// }


/*
 * utility function to implement path join in python
 * 
 * @param dir  : directory to combine
 * @param file : file name
 */
char* path_join(const char *dir, const char* file)
{
    int size1 = strlen(dir);
    int size2 = strlen(file);

    if (size1 == 0 || size2 == 0) return NULL;
    
    char *buf = (char*)malloc( (size1 + size2 + 2) * sizeof(char));
    if (buf == NULL) return NULL;

    strcpy(buf, dir);

    if (dir[size1-1] != '/')
    {
        strcat(buf, PATH_JOIN_SEP);
    }

    if (file[0] == '/')
    {
        strcat(buf, file+1);
    }
    else
    {
        strcat(buf, file);
    }

    return buf;
}




/*
 * free the memory of params in convex clustering
 * 
 * @param p: the ptr to cvx_clustr_param
 */
void param_free(cvx_clustr_param *param)
{
    sp_matrix_free(param->A);
    free(param->e_r);
    free(param->e_c);
    free(param->x);
    free(param->D);
    free(param->w_c);
    free(param->w_r);
}



void darray_set_all(double *v, double val, int n)
{
    unsigned int j;
    for (j = 0; j < n; ++j)
    {
        v[j] = val;
    }
}


/* Polar (Box-Mueller) method; See Knuth v2, 3rd ed, p122 */
double random_gaussian(const double sigma)
{
    double x, y, r2;

    do
    {
        /* choose x,y in uniform square (-1,-1) to (+1,+1) */
        x = -1 + 2 * ( (double)(rand()) + 1. )/( (double)(RAND_MAX) + 1. );
        y = -1 + 2 * ( (double)(rand()) + 1. )/( (double)(RAND_MAX) + 1. );
        
        /* see if it is in the unit circle */
        r2 = x * x + y * y;
    } while ( r2 > 1.0 || r2 == 0);

    /* Box-Muller transform */
    return sigma * y * sqrt(-2.0 * log (r2) / r2);
}


int* prox(double *v, double gamma_c, double gamma_r, double *w_c, double *w_r, double tau,
    int E_c, int E_r, int n, int p)
{
    int offset = E_c * p;
    int *edge_ind = (int *) malloc((E_c + E_r) * sizeof(int));
    
    // printf("E_c = %d, E_r = %d\n", E_c, E_r);    

    #pragma omp parallel
    {
        double thres;
        int j;

        #pragma omp for
        for (j = 0; j < E_c; ++j)
        {
            thres = gamma_c * w_c[j] / tau;
            edge_ind[j] = thres / cblas_dnrm2(p, v+j*p, 1) >= 1 ? 1 : 0;
            // thres = 1 - thres / cblas_dnrm2(p, v+j*p, 1);
            // cblas_dscal(p, max(0,thres), v+j*p, 1);
        }

        #pragma omp for
        for (j = 0; j < E_r; ++j)
        {
            thres = gamma_r * w_r[j] / tau;
            edge_ind[E_c+j] = thres / cblas_dnrm2(n, v+offset+j, E_r) >= 1 ? 1 : 0;
            // thres = 1 - thres / cblas_dnrm2(n, v+offset+j, E_r);
            // cblas_dscal(n, max(0,thres), v+offset+j, E_r);
        }

    }

    return edge_ind;
};


void create_graph(int *ind, igraph_t *r_g, igraph_t *c_g, edge *e_r, edge *e_c, int E_r, int E_c)
{
    int j;
    unsigned int offset = E_c;
    
    for (j = 0; j < E_c; ++j)
    {
        if (ind[j] == 1)
        {
            igraph_add_edge(c_g, e_c[j]._from, e_c[j]._to);
            igraph_add_edge(c_g, e_c[j]._to, e_c[j]._from);
        }
    }

    for (j = 0; j < E_r; ++j)
    {
        if (ind[offset+j] == 1)
        {
            igraph_add_edge(r_g, e_r[j]._from, e_r[j]._to);
            igraph_add_edge(r_g, e_r[j]._to, e_r[j]._from);
        }
    }

}


void double_vector_fprintf(FILE *stream, double *v, int n)
{
    int j;
    for (j = 0; j < n; ++j)
    {
        fprintf(stream, "%g\n", v[j]);
    }
}
