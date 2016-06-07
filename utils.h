#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include <stdlib.h>

#define ROOT 0
#define ARRAY_IND(i, j, cols_no) (i*cols_no + j)

typedef struct sparse_type_s {
    int rows_no, cols_no;
    int *IA, *JA;
    double *A;
} sparse_type;

typedef struct dense_type_s {
    double *vals;
    int cols_no, rows_no;
} dense_type;

void alloc_sparse(sparse_type *st, int nnz, int rows_no, int cols_no);
void free_sparse(sparse_type *st);
void parse_csr(FILE *fd, sparse_type *st);
void alloc_dense(int col_no, int rows_no, dense_type *m);
void free_dense(dense_type *m);
void copy_dense(dense_type *source, dense_type *destination);
void DEBUG_ALL_SPARSES(int num_processes, sparse_type *msgs);
void DEBUG_SPARSE(int k, sparse_type *msg);
void DEBUG_DENSE(int mpi_rank, dense_type *d);
void DEBUG_SPARSE_TO_DENSE(int k, int num_processes, int sub_size, sparse_type *msg);

#endif
