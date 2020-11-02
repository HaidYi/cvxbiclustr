
#include "cvxclustr.h"



void cvxclustr(cvx_clustr_param *param, cvx_clustr_output *out, char *solver,
    int max_iter, double tol, double lr, bool verbose)
{
    
    double *lambda0 = (double *) malloc((param->E_c * param->p + param->E_r * param->n) * sizeof(double));
    darray_set_all(lambda0, 0.0, param->E_c * param->p + param->E_r * param->n);
    
    /* <<< Initialize variables needed */
    solver_arg args = {verbose, max_iter, tol, lr};

    if (!strcmp(solver, "fasta"))
    {
        // transpose the matrix
        sp_matrix_transpose(param->A);
        
        fastaOpt opt;
        setFastaOpt(&opt, param->A, param->x, param->D, &args);

        out->v_sol = fasta(param->A, param->x, lambda0, param->w_c, param->w_r, param->D,
              param->gamma_c, param->gamma_r, param->n, param->p, param->E_c, param->E_r, &opt, out);
    }
    else if (!strcmp(solver, "pgd"))
    {
        pgdOpt opt;
        setPgdOpt(&opt, &args);
        
        out->v_sol = pgd(param->A, param->x, lambda0, param->w_c, param->w_r, 
             param->gamma_c, param->gamma_r, param->n, param->p, param->E_c, param->E_r, &opt, out);
    }
    else
    {
        printf("error print\n");
        char msg[MAX_STR_LENGTH];
        sprintf(msg, "Unknown solver: %s", solver);
        ERROR_INFO(msg);
    }

    // get the connected vec
    int *edge_ind = prox(out->v_sol, param->gamma_c, param->gamma_r, param->w_c, param->w_r,
        out->tau, param->E_c, param->E_r, param->n, param->p);
    
    // generate clustering asignment
    igraph_t v_graph_rows, v_graph_cols;

    // allocate memory
    // igraph_vector_init (&(out->row_memship), param->p);
    // igraph_vector_init (&(out->col_memship), param->n);
    // igraph_vector_init (&(out->csize_row), param->p);
    // igraph_vector_init (&(out->csize_col), param->n);

    igraph_empty (&v_graph_rows, param->p, IGRAPH_UNDIRECTED);
    igraph_empty (&v_graph_cols, param->n, IGRAPH_UNDIRECTED);

    create_graph(edge_ind    ,
                &v_graph_rows,
                &v_graph_cols,
                param->e_r   ,
                param->e_c   ,
                param->E_r   ,
                param->E_c);
    
    igraph_clusters(&v_graph_cols, &(out->col_memship), &(out->csize_col), &(out->no_col), IGRAPH_STRONG);
    igraph_clusters(&v_graph_rows, &(out->row_memship), &(out->csize_row), &(out->no_row), IGRAPH_STRONG);

    /* >>> free intermediate variable */
    
    free(lambda0);
    free(edge_ind);
    free(out->v_sol);
    
    igraph_destroy (&v_graph_rows);
    igraph_destroy (&v_graph_cols);
    
}


