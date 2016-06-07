#ifndef ZAD2_INNERCOLUMN_H
#define ZAD2_INNERCOLUMN_H

#include "utils.h"
#include "mpi.h"

void broadcast_sparse(int mpi_rank, int num_processes, int columns_no, sparse_type* sparse);
void generate_dense(dense_type *dense, int rows_no, int cols_no, int mpi_rank, int num_processes, int gen_seed);
void get_sparse_part(int mpi_rank, int num_processes, sparse_type* sparse, sparse_type* Apart);
void split_comm(int mpi_rank, int repl_fact, int num_processes, MPI_Comm *sub_comm, int *sub_rank, int *sub_size);
void broadcast_A_parts_in_groups(int mpi_rank, sparse_type *As, sparse_type *A, int sub_size, MPI_Comm *sub_comm);
void join_sparse_type(int mpi_rank, int sub_size, int num_processes, sparse_type *As, sparse_type *A);
void compute_matrix(int exponent, int sub_size, int num_processes, int mpi_rank, int all_cols_no,
                    dense_type *C, sparse_type *As, dense_type *B);
void gather_and_show_results(int mpi_rank, int num_processes, int all_cols_no, int rows_no, dense_type *part);
void count_and_print_ge_elements(int mpi_rank, int num_processes, dense_type* C, double ge_elm);
#endif //ZAD2_INNERCOLUMN_H
