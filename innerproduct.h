

#ifndef ZAD2_INNERPRODUCT_H
#define ZAD2_INNERPRODUCT_H

#include "utils.h"

void get_sparse_part_inner(int mpi_rank, int num_processes, sparse_type* sparse, sparse_type* Apart);
void broadcast_A_B_parts_in_groups(int mpi_rank, sparse_type *As, sparse_type *A, dense_type *Bs, dense_type *B,
                                   int sub_size, MPI_Comm *sub_comm);
void join_sparse_type_inner(int mpi_rank, int sub_size, int num_processes, sparse_type *As, sparse_type *A);
void join_dense_type_inner(int mpi_rank, int sub_size, int num_processes, dense_type *Bs, dense_type *B);
void compute_matrix_inner(int exponent, int sub_size, int num_processes, int mpi_rank, int all_cols_no, MPI_Comm *comm,
                            dense_type *C, sparse_type *As, dense_type *B);
void gather_and_show_results_inner(int mpi_rank, int sub_size, int num_processes, dense_type *C);
void count_and_print_ge_elements_inner(int mpi_rank, int num_processes, int sub_size, dense_type *C, double ge_elm);

#endif //ZAD2_INNERPRODUCT_H
