#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <assert.h>
#include <getopt.h>

#include "densematgen.h"
#include "utils.h"
#include "blockedcolumn.h"
#include "innerproduct.h"

int main(int argc, char * argv[])
{
    int show_results = 0;
    int use_inner = 0;
    int gen_seed = -1;
    int repl_fact = 1;

    int option = -1;

    double comm_start = 0, comm_end = 0, comp_start = 0, comp_end = 0;
    int num_processes = 1;
    int mpi_rank = 0;
    int exponent = 1;
    double ge_element = 0;
    int count_ge = 0;
    int columns_no;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    sparse_type* sparse;
    sparse = malloc(sizeof(sparse_type)); //TODO free
    while ((option = getopt(argc, argv, "vis:f:c:e:g:")) != -1) {
        switch (option) {
            case 'v': show_results = 1;
                break;
            case 'i': use_inner = 1;
                break;
            case 'f':
                if ((mpi_rank) == ROOT) {
                    FILE *fd = fopen(optarg, "r");
                    parse_csr(fd, sparse);
                    columns_no = sparse->cols_no;
                    close(fd);
                }
                break;
            case 'c': repl_fact = atoi(optarg);
                break;
            case 's': gen_seed = atoi(optarg);
                break;
            case 'e': exponent = atoi(optarg);
                break;
            case 'g': count_ge = 1;
                ge_element = atof(optarg);
                break;
            default: fprintf(stderr, "error parsing argument %c exiting\n", option);
                MPI_Finalize();
                return 3;
        }
    }
    if ((gen_seed == -1) || ((mpi_rank == 0) && (sparse == NULL)))
    {
        fprintf(stderr, "error: missing seed or sparse matrix file; exiting\n");
        MPI_Finalize();
        free_sparse(sparse);
        return 3;
    }


    comm_start = MPI_Wtime();
    sparse_type A;
    dense_type B, C;

    if (use_inner) {
        get_sparse_part_inner(mpi_rank, num_processes, sparse, &A);
        columns_no = A.cols_no;
    } else {
        get_sparse_part_blocked(mpi_rank, num_processes, sparse, &A);
        //set columns size
        MPI_Bcast(&columns_no, 1, MPI_INT, ROOT, MPI_COMM_WORLD);
    }
    //generate dense
    if (use_inner) {
        generate_dense(&B, A.cols_no, A.rows_no, mpi_rank, num_processes, gen_seed);
    } else {
        generate_dense(&B, A.rows_no, A.cols_no, mpi_rank, num_processes, gen_seed);
    }

    //divide processes into groups
    MPI_Comm sub_comm;
    int sub_rank, sub_size;
    split_comm(mpi_rank, repl_fact, num_processes, &sub_comm, &sub_rank, &sub_size);

    ///broadcast in groups
    sparse_type *As = malloc(sub_size * sizeof(sparse_type)); //TODO free
    dense_type *Bs;
    if (use_inner) {
        Bs = malloc(sub_size * sizeof(sparse_type)); //TODO free
        broadcast_A_B_parts_in_groups(mpi_rank, As, &A, Bs, &B, sub_size, &sub_comm);
    } else {
        broadcast_A_parts_in_groups(mpi_rank, As, &A, sub_size, &sub_comm);
    }

    sparse_type A_joined;
    dense_type B_joined;
    if (use_inner) {
        join_sparse_type_inner(mpi_rank, sub_size, num_processes, As, &A_joined);
        join_dense_type_inner(mpi_rank, sub_size, num_processes, Bs, &B_joined);
    } else {
        join_sparse_type_blocked(mpi_rank, sub_size, num_processes, As, &A_joined);
    }

    comm_end = MPI_Wtime();
    comp_start = MPI_Wtime();

    if (use_inner) {
        compute_matrix_inner(exponent, sub_size, num_processes, mpi_rank, columns_no, &sub_comm, &C, &A_joined, &B_joined);
    } else {
        compute_matrix_blocked(exponent, sub_size, num_processes, mpi_rank, columns_no, &C, &A_joined, &B);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    comp_end = MPI_Wtime();

    MPI_Barrier(MPI_COMM_WORLD);

    if (show_results)
    {
        if (use_inner) {
            gather_and_show_results_inner(mpi_rank, sub_size, num_processes, &C);
        } else {
            gather_and_show_results(mpi_rank, num_processes, columns_no, C.rows_no, &C);
        }
    }
    if (count_ge)
    {
        if (use_inner) {
            count_and_print_ge_elements_inner(mpi_rank, num_processes, sub_size, &C, ge_element);
        } else {
            count_and_print_ge_elements(mpi_rank, num_processes, &C, ge_element);
        }
    }

    MPI_Comm_free(&sub_comm);
    MPI_Finalize();
    return 0;
}