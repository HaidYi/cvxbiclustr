/* Openmp implementation of Convex Co-clustering (Omp-CoCo)
 *
 * This project is under the supervision of Professor Eric Chi at NCSU.
 *  
 * The implementation uses OpenMP for parallel computing
 * and OpenBlas for linear algebra
 *
 * @author: Haidong Yi, haidyi@cs.unc.edu
*/

#include <iostream>
using namespace std;
#include <map>
#include <list>
#include <argp.h>

#include <cvxclustr/cvxclustr.h>
#include "mmio.h"


typedef std::pair<int, int> _edge;

typedef struct {
    bool verbose;
    unsigned int max_iter, n_threads;
    double tol, lr;
    char *gamma, *solver, *dirX, *dirGr, *dirGc, *dir_soln;
} arguments;

void update_x(double *x, int *D, double *x_new, int *D_new, igraph_vector_t col_mem, igraph_vector_t row_mem, 
    igraph_vector_t csize_col, igraph_vector_t csize_row, int no_col, int no_row, int n, int p);
void update_e(double *w_c, double *w_r, edge *e_c, edge *e_r, igraph_vector_t col_mem, igraph_vector_t row_mem, int *E_c, int *E_r);
void param_update(cvx_clustr_param *param, cvx_clustr_output *out);
void out_reallocate(cvx_clustr_param *param, cvx_clustr_output *out);

// io part
void param_show(arguments *args);
int param_read_size(FILE*, int*);
int mm_read_vec_array(FILE *f, double *v, int len);
int read_edge_list(edge **e, double **w, int *num_e, char *dir, int *flag);
int edge_list_fscanf(FILE *stream, edge **e, double **w, int *num_e);
void edgelist_to_A(sp_matrix *A, edge *e_c, int E_c, edge *e_r, int E_r, int p, int n);

void int_vector_fprintf(FILE *stream, int *v, int n);

/* program documentation */
const char *argp_program_version = "Version: 0.01";
const char *argp_program_bug_address = "haidyi@cs.unc.edu";
static char doc[] = "The parallel implementation of Convex Co-clustering with OpenMP.";
static char args_doc[] = "<matrixfile>"; // add more instructions here


/* argument options */
static struct argp_option options[] = {
    { 0,  0,  0, 0, "Basic options:"},
    { 0, 'h', 0, 0, "Show brief help on version and usage"},

    { 0,  0,  0, 0, "Options controlling hyperparameters:"},
    { "gamma", 'g', "<str>", 0, ": Parameter of penalty params (comma seperated)"},

    { 0,  0,  0, 0, "Options controlling input file:"},
    { "rgfile", 'R', "<file>", 0, ": Path to the row graph file"},
    { "cgfile", 'C', "<file>", 0, ": Path to the column graph file"},
    { 0, 'o', "<file>", 0, ": Path to solution file (co-clustering)"},

    { 0,  0,  0, 0, "Options controlling optimization:"},
    { "solver", 's', "<str>", 0, ": optim solver to use"},
    { "nthreads", 'p', "<int>", 0, ": Number of threads to use"},
    { "max_iter", 'm', "<int>", 0, ": Max iterations to run"},
    { "tol", 't', "<double>", 0, ": Tolerance of convergence"},
    { "lr", 'l', "<double>", 0, ": Learning rate"},
    { "verbose", 'v', 0, 0, ": Produce verbose output"},

    { 0, 0, 0, 0, "Options controlling help, usage and version"},
    { 0 }
};


static error_t parse_opt(int key, char *arg, struct argp_state *state) {
    
    arguments *args = static_cast<arguments*>(state->input);
    
    switch (key) {
        case 'h':
            argp_usage (state);
        case 'g':
            args->gamma = arg;
            break;
        case 'R':
            args->dirGr = arg;
            break;
        case 'C':
            args->dirGc = arg;
            break;
        case 'o':
            args->dir_soln = arg;
            break;
        case 's':
            args->solver = arg;
            break;
        case 'p':
            args->n_threads = arg ? atoi(arg): 1;
            break;
        case 'm':
            args->max_iter = arg ? atoi(arg): 1000;
            break;
        case 'l':
            args->lr = arg ? atof(arg): 1e-3;
            break;
        case 't':
            args->tol = arg ? atof(arg): 1e-3;
            break;
        case 'v':
            args->verbose = true;
            break;
        case ARGP_KEY_NO_ARGS:
            argp_usage (state);
        case ARGP_KEY_ARG:
            args->dirX = arg;
            return 0;
        default:
            return ARGP_ERR_UNKNOWN;
    }
    return 0;
}

static struct argp argp = { options, parse_opt, args_doc, doc };


int main(int argc, char **argv)
{
    /* set default configs */
    arguments args = {
        true, 1000, 1, 1e-3, 1e-3, (char *)"1.0", (char *)"fasta", (char*)"", (char *)"", (char *)"", (char *)""
    };
    
    /* parse command line options and display */
    argp_parse(&argp, argc, argv, 0, 0, &args);
    param_show(&args);

    // parse gamma list
    std::list<double> gammas;
    char buf[MAX_STR_LENGTH];
    
    strcpy(buf, args.gamma);
    const char *split = ",";
    char *gamma_i; char *_buf = buf;
    while ((gamma_i = strsep(&_buf, split)) != NULL)
    {
        gammas.push_back(atof(gamma_i));
    }
    
    /* >>> read matrix X */
    sp_matrix A; A.sptype = SPMAT_CSR;
    cvx_clustr_param param = {
      0, 0, 0, 0, 0, 0, &A, NULL, NULL, NULL, NULL, NULL 
    };
    
    // X: p x n (feature x sample)
    FILE *f;

    f = fopen(args.dirX, "r");
    if (f == NULL)
    {
        ERROR_INFO("Error: open matrix file");
        NOT_FOUND;
    }
    mm_read_mtx_array_size(f, &param.p, &param.n);

    param.x = (double *) malloc(param.p * param.n * sizeof(double));
    param.D = (int    *) malloc(param.p * param.n * sizeof(int));
        
    int j;
    for (j = 0; j < param.p * param.n; ++j)
    {
      param.D[j] = 1;
    }
    mm_read_vec_array(f, param.x, param.n * param.p);
    fclose(f);

    /* <<< read matrix x */
    
    /* >>> read the edge of row and column graphs */
    int r_flag = 0, c_flag = 0;

    if (read_edge_list(&(param.e_r), &(param.w_r), &param.E_r, args.dirGr, &r_flag) || 
        read_edge_list(&(param.e_c), &(param.w_c), &param.E_c, args.dirGc, &c_flag))
    {
        free(param.x);
        NOT_IMPLEMENTED;
    }
    
    if (r_flag && c_flag) // check input
    {
        free(param.x);
        ERROR_INFO("Error: at least one graph as input");
        NOT_FOUND;
    }

    // generate sparse matrix A with edge list
    edgelist_to_A(param.A, param.e_c, param.E_c, param.e_r, param.E_r, param.p, param.n);
    
    /* <<< read the edge of row and column graphs */

    /* >>> Initialize variables needed */
    cvx_clustr_output out;
    
    out.u_sol = (double *) malloc(param.n * param.p * sizeof(double));   // init soln variable
    cblas_dcopy(param.n * param.p, param.x, 1, out.u_sol, 1);
    out.obj = (double *) malloc((args.max_iter + 1) * sizeof(double)); // objFuncEval record

    igraph_vector_init (&(out.col_memship), param.n);
    igraph_vector_init (&(out.row_memship), param.p);
    igraph_vector_init (&(out.csize_col), param.n);
    igraph_vector_init (&(out.csize_row), param.p);
    
    /* <<< Initialize variables needed */

    // run the optimization and time it
    LOG_INFO("start optimization...");
    omp_set_num_threads(args.n_threads);

    FILE *handler = fopen(args.dir_soln, "a");
    if (handler == NULL)
    {
        ERROR_INFO("Output file cannot be found, exit...");
    }
    
    double start_time = omp_get_wtime();
    
    std::list<double>::iterator it;
    for (it = gammas.begin(); it != gammas.end(); ++it)
    {
        fprintf(handler, "gamma: %g\n", *it);
        // update gamma
        param.gamma_c = *it;
        param.gamma_r = *it;

        // run convex (bi)-clustering
        // double_vector_fprintf(stdout, param.x, param.n * param.p);
        // printf("\n");
        // int_vector_fprintf(stdout, param.D, param.n * param.p);            
        // printf("problem size: n: %d p: %d\n", param.n, param.p);        

        cvxclustr(&param, &out, args.solver, args.max_iter, args.tol, args.lr, args.verbose);
       
        LOG_INFO("Writing Result ...");
        // double_vector_fprintf(handler, out.u_sol, param.n * param.p);
        
        //fprintf(handler, "col_memship:\n");
        //igraph_vector_fprint(&(out.col_memship), handler);
        //fprintf(handler, "row_memship:\n");
        //igraph_vector_fprint(&(out.row_memship), handler);        
 
        // update parameters for next interation
        param_update(&param, &out);
        out_reallocate(&param, &out);

    }

    double end_time = omp_get_wtime();
    fprintf(stdout, "run_time: %.4f\n", end_time - start_time);

    /* close file stream */
    fclose(handler);

    /* <<< free memory */
    param_free(&param);

    igraph_vector_destroy (&(out.col_memship));
    igraph_vector_destroy (&(out.row_memship));
    igraph_vector_destroy (&(out.csize_col));
    igraph_vector_destroy (&(out.csize_row));

    free(out.u_sol);
    free(out.obj);

    return 0;
}


void update_x(double *x, int *D, double *x_new, int *D_new, igraph_vector_t col_mem, igraph_vector_t row_mem, 
    igraph_vector_t csize_col, igraph_vector_t csize_row, int no_col, int no_row, int n, int p)
{
    int i, j;
    int idx;

    // double *x_new = (double *) malloc(no_col * no_row * sizeof(double));
    
    for (i = 0; i < no_col; ++i)
    {
        for (j = 0; j < no_row; ++j)
        {
            x_new[i * no_row + j] = 0;
            D_new[i * no_row + j] = 0;
        }
    }

    for (j = 0; j < n; ++j)
    {
        for (i = 0; i < p; ++i)
        {
            idx = VECTOR(col_mem)[j] * no_row + VECTOR(row_mem)[i];
            x_new[idx] += x[j*p+i] * D[j*p+i];
            D_new[idx] += D[j*p+i];
        }
    }

    for (j = 0; j < no_col; ++j)
    {
        for (i = 0; i < no_row; ++i)
        {
            idx = j * no_row + i;
            x_new[idx] /= D_new[idx];
        }
    }


    free(x);   // free the memory
    free(D);

}


void update_e(double *w_c, double *w_r, edge *e_c, edge *e_r, igraph_vector_t col_mem, igraph_vector_t row_mem, int *E_c, int *E_r)
{
    std::map<_edge, double> edge_to_wts;

    int j;
    int _from, _to;
    
    for (j = 0; j < *E_c; ++j)
    {
        _from = VECTOR(col_mem)[e_c[j]._from];
        _to   = VECTOR(col_mem)[e_c[j]._to];
        
        if (_from != _to)
        {
            if (edge_to_wts.find(std::make_pair(_from, _to)) != edge_to_wts.end())
            {
                edge_to_wts[std::make_pair(_from, _to)] += w_c[j];
            }
            else
            {
                edge_to_wts.insert(map<_edge, double>::value_type(std::make_pair(_from, _to), w_c[j]));
            }
        }
    }

    *E_c = edge_to_wts.size();
    
    // create edge list and wts from map
    std::map<_edge, double>::iterator it; 
    
    it = edge_to_wts.begin();
    j = 0;

    for (; it!=edge_to_wts.end(); ++it)
    {
        e_c[j]._from = it->first.first; 
        e_c[j]._to = it->first.second;
        w_c[j] = it->second;

        ++j;
    }
    
    
    edge_to_wts.clear(); // clear map

    for (j = 0; j < *E_r; ++j)
    {
        _from = VECTOR(row_mem)[e_r[j]._from];
        _to   = VECTOR(row_mem)[e_r[j]._to];

        if (_from != _to)
        {
            if (edge_to_wts.find(std::make_pair(_from, _to)) != edge_to_wts.end())
            {
                edge_to_wts[std::make_pair(_from, _to)] += w_r[j];
            }
            else
            {
                edge_to_wts.insert(map<_edge, double>::value_type(std::make_pair(_from, _to), w_r[j]));
            }
        }

    }

    *E_r = edge_to_wts.size();
    
    // create edge list and wts from map
    it = edge_to_wts.begin();
    j = 0;

    for (; it!=edge_to_wts.end(); ++it)
    {
        e_r[j]._from = it->first.first; 
        e_r[j]._to = it->first.second;
        w_r[j] = it->second;

        ++j;
    }

    edge_to_wts.clear();

}


void param_update(cvx_clustr_param *param, cvx_clustr_output *out)
{
    // calculate compressed x and update parameters
    double *x_new = (double *) malloc(out->no_col * out->no_row * sizeof(double));
    int    *D_new = (int    *) malloc(out->no_col * out->no_row * sizeof(int));
    update_x(param->x, param->D, x_new, D_new, out->col_memship, out->row_memship, out->csize_col, out->csize_row, out->no_col, out->no_row, param->n, param->p);
    param->x = x_new;
    param->D = D_new;

    update_e(param->w_c, param->w_r, param->e_c, param->e_r, out->col_memship, out->row_memship, &(param->E_c), &(param->E_r));

    // update feature num and sample num
    param->n = out->no_col; 
    param->p = out->no_row;

    // construct new sp_matrix A
    sp_matrix_free(param->A);
    edgelist_to_A(param->A, param->e_c, param->E_c, param->e_r, param->E_r, param->p, param->n);
    
}


void out_reallocate(cvx_clustr_param *param, cvx_clustr_output *out)
{

    free(out->u_sol);
    out->u_sol = (double *) malloc(param->n * param->p * sizeof(double));
    cblas_dcopy(param->n * param->p, param->x, 1, out->u_sol, 1);

    igraph_vector_destroy (&(out->col_memship));
    igraph_vector_destroy (&(out->row_memship));

    igraph_vector_init (&(out->col_memship), param->n);
    igraph_vector_init (&(out->row_memship), param->p);

    igraph_vector_destroy (&(out->csize_col));
    igraph_vector_destroy (&(out->csize_row));

    igraph_vector_init (&(out->csize_col), param->p);
    igraph_vector_init (&(out->csize_row), param->n);
    
}


void param_show(arguments *args)
{
    char config[] ={
                    "--------------configuration--------------\n"
                    "input matrix file:\t%s\n"
                    "input row graph:\t%s\n"
                    "input column file:\t%s\n"
                    "output matrix file:\t%s\n"
                    "gamma: %s\n"
                    "lr: %g\n"
                    "max_iter: %d\n"
                    "#of threads: %d\n"
                    "optimization precision: %g\n"
                    "-----------------------------------------\n"
                    };

    printf(config, args->dirX, args->dirGr, args->dirGc, args->dir_soln,
           args->gamma, args->lr, args->max_iter, args->n_threads, args->tol);
}




/*
 * Implement a util function to read number of edges
 * 
 * 
 * @param f: FILE Pointer
 * @param E: int to store n_edges
 *
*/
int param_read_size(FILE *f, int *E)
{    
    char line[MM_MAX_LINE_LENGTH];
    int num_items_read;
    /* set 0 values in case we exit with errors */
    *E = 0;

    /* scanning unitl reach the end-of-comments */
    do {
        if (fgets(line, MM_MAX_LINE_LENGTH, f) == NULL)
            return MM_PREMATURE_EOF;
    } while(line[0] == '%');

    if (sscanf(line, "%d", E) == 1)
        return 0;
    
    else /* blank line */
    do {
        num_items_read = fscanf(f, "%d", E);
        if (num_items_read == EOF) return MM_PREMATURE_EOF;
    }
    while (num_items_read != 1);

    return 0;
}


/*
 * Reading the vector file (matrix market type)
 *
 * Note: calling this function after mm_read_mtx_array_size()
 * 
 * @param f: FILE pointer
 * @param v: vector to set
*/
int mm_read_vec_array(FILE *f, double *v, int len)
{
    double entry;
    int i;
    
    for (i = 0; i < len; i++) {
        fscanf(f, "%lf", &entry);
        v[i] = entry;
    }
    // fclose(f);

    return 0;
}


/*
 * Reading the vector file (matrix market type)
 *
 * Note: calling this function after mm_read_mtx_array_size()
 * 
 * @param f: FILE pointer
 * @param arr: array to set
*/
int mm_read_array(FILE *f, double *arr, int N)
{
    double entry;
    int i;
    
    for (i = 0; i < N; i++) {
        fscanf(f, "%lf", &entry);
        arr[i] = entry;
    }
    // fclose(f);

    return 0;
}


/*
 * Reading the edge list (sparse matrix io format)
 * 
 * 
 * 
 * @param stream : FILE ptr
 * @param e      : edge array
 * @param w      : weight array
 */
int edge_list_fscanf(FILE *stream, edge **e, double **w, int *num_e)
{
    unsigned int size1, size2, ne;
    char buf[MAX_STR_LENGTH];
    int found_header = 0;

    while (fgets(buf, MAX_STR_LENGTH, stream) != NULL)
    {
        int c;

        /* skip comments */
        if (*buf == '%')
            continue;

        c = sscanf(buf, "%u %u %u", &size1, &size2, &ne);
        if (c == 3)
        {
            found_header = 1;
            break;
        }
    }

    if (!found_header || size1 != size2)
    {
        ERROR_MSG("fscanf failed reading header");
    }

    // allocate memory for reading edges
    *e = (edge   *) malloc(sizeof(edge)   * ne);
    *w = (double *) malloc(sizeof(double) * ne);
    *num_e = ne;

    unsigned int i, j, k;
    double entity;

    for (k = 0; k < ne; k++)
    {
        if (fgets(buf, 1024, stream) != NULL)
        {
            int c = sscanf(buf, "%u %u %lf", &i, &j, &entity);

            if (c < 3)
            {
                // error occurs, so free memory allocated in this func
                free(e);
                free(w);
                return EXIT_FAILURE;
            }
            else
            {
                (*e)[k]._from = i-1;
                (*e)[k]._to   = j-1;
                (*w)[k]       = entity;
            }
        }
    }

    return 0;
}

/*
 * read the sparse graph using matrix market format
 * 
 * 
 * @param e    : edge array
 * @param w    : weights array
 * @param dir  : if dir is empty (check input) flag <- 1 else 0
 * @param flag : flag <- 1 if dir == empty else 0
 * 
 * @return: 0: success 1: failure
 */ 
int read_edge_list(edge **e, double **w, int *num_e, char *dir, int *flag)
{
    if (strcmp(dir, ""))
    {
        FILE *f = fopen(dir, "r");
        if (f != NULL)
        {
            if (edge_list_fscanf(f, e, w, num_e))
            {
                fclose(f);
                return EXIT_FAILURE;
            }
            fclose(f);
        }
        else
        {
            char msg[MAX_STR_LENGTH];
            sprintf(msg, "ERROR: %s does not exist, exiting.", dir);
            ERROR_INFO(msg);
            return EXIT_FAILURE;
        }
    } 
    else
    {
        *flag = 1; // set flag
        char msg[MAX_STR_LENGTH];
        sprintf(msg, "Warning: No input file %s", dir);
        WARN_INFO(msg);
    }

    return 0;
}


/*
 * Reading the matrix file (matrix market type)
 *
 * Note: calling this function after mm_read_mtx_array_size()
 * 
 * @param f: FILE pointer
 * @param M: matrix to set
*/
// int mm_read_mtx_array(FILE *f, gsl_matrix *M)
// {
//     int m = M->size1;
//     int n = M->size2;
//     double entry;
//     int row, col;

//     for (int i = 0; i < m * n; i++) {
//         row = i % m;
//         col = floor(i/m);
//         fscanf(f, "%lf", &entry);
//         gsl_matrix_set(M, row, col, entry);
//     }
//     // fclose(f);

//     return 0;
// };


void edgelist_to_A(sp_matrix *A, edge *e_c, int E_c, edge *e_r, int E_r, int p, int n)
{
    A->m = E_c * p + E_r * n;
    A->n = n * p;
    A->sptype = SPMAT_CSR;
    A->nz = A->m * 2 ;

    A->p = (unsigned int *) malloc((A->m + 1) * sizeof(unsigned int));
    A->p[0] = 0;

    A->i = (unsigned int *) malloc((2*E_c*p + 2*E_r*n) * sizeof(unsigned int));
    A->data = (double *) malloc((2*E_c*p + 2*E_r*n) * sizeof(double));

    unsigned int j, k;
    unsigned int idx, offset;
    
    for(j = 0; j < E_c; ++j)
    {
        for(k = 0; k < p; ++k)
        {
            idx = j * p + k;

            A->data[2*idx  ] =  1;
            A->data[2*idx+1] = -1;

            A->i[2*idx  ] = e_c[j]._from * p + k;
            A->i[2*idx+1] = e_c[j]._to * p + k;

            A->p[idx+1] = A->p[idx] + 2;   
        }
    }

    offset = E_c * p;

    for (j = 0; j < n; ++j)
    {
        for (k = 0; k < E_r; ++k)
        {
            idx = E_r * j + k;

            A->data[2*offset+2*idx  ] =  1;
            A->data[2*offset+2*idx+1] = -1;

            A->i[2*offset+2*idx  ] = j*p + e_r[k]._from;
            A->i[2*offset+2*idx+1] = j*p + e_r[k]._to;

            A->p[offset+idx+1] = A->p[offset+idx] + 2;
        }
    }

};


void int_vector_fprintf(FILE *stream, int *v, int n)
{
    int j;
    for (j = 0; j < n; ++j)
    {
        fprintf(stream, "%d\n", v[j]);
    }
};
