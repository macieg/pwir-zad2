#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define ROOT 0
#define ARRAY_IND(i, j, cols_no) ((i)*(cols_no) + j)

#define A_TAG 0
#define JA_TAG 1
#define IA_TAG 2
#define COLS_NO_TAG 3
#define ROWS_NO_TAG 3

typedef struct sparse_type_s {
    int rows_no, cols_no;
    int *IA, *JA;
    double *A;
} sparse_type;

typedef struct dense_type_s {
    double *vals;
    int cols_no, rows_no;
} dense_type;

int proc_to_cols_no(int mpi_rank, int num_processes, int cols_no);
void send_sparse_async(MPI_Request *req, int desination, sparse_type *msg);
void receive_sparse_rows_no(int source, int rows_no, sparse_type *msg);
void send_sparse(int desination, sparse_type *msg);
void receive_sparse(int source, sparse_type *msg);
void alloc_sparse(sparse_type *st, int nnz, int rows_no, int cols_no);
void free_sparse(sparse_type *st);
int col_part_to_col(int col_no, int mpi_rank, int num_processes, int cols_no);
void generate_dense(dense_type *dense, int rows_no, int cols_no, int mpi_rank, int num_processes, int gen_seed);
void split_comm(int mpi_rank, int repl_fact, int num_processes, MPI_Comm *sub_comm, int *sub_rank, int *sub_size);
void parse_csr(FILE *fd, sparse_type *st);
void alloc_dense(int col_no, int rows_no, dense_type *m);
void free_dense(dense_type *m);
void copy_dense(dense_type *source, dense_type *destination);
void DEBUG_ALL_SPARSES(int num_processes, sparse_type *msgs);
void DEBUG_SPARSE(int k, sparse_type *msg);
void DEBUG_SPARSE_CSR(int k, sparse_type *msg);
void DEBUG_DENSE(int mpi_rank, dense_type *d);
void DEBUG_SPARSE_TO_DENSE(int k, int num_processes, int sub_size, sparse_type *msg);

#endif
