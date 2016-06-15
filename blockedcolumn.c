#include <stdlib.h>

#include "blockedcolumn.h"
#include "utils.h"
#include "mpi.h"
#include "densematgen.h"

//dla danej kolumny z calej macierzy A zwraca indeks procesu
inline int col_to_proc(int col_no, int num_processes, int cols_no) {
    int cpp = cols_no / num_processes; //podstawa
    int more = cols_no % num_processes; //liczba procesow ktorym dodamy 1 (pierwsze z brzegu)

    int procmorecols = more * (cpp + 1); //liczba kolumn zabranych przez procesy z wieksza iloscia kolumn
    if (col_no <= procmorecols) { //kolumna w procesie z wieksza iloscia kolumn
        return (col_no) / (cpp + 1);
    } else { //kolumna w procesie z mniejsza iloscia kolumn
        return more + (col_no - procmorecols) / (cpp);
    }
}

//dla danej kolumny z calej macierzy zwraca jej indeks w macierzy czesciowej
inline int col_to_col_part(int col_no, int num_processes, int glob_cols_no) {
    int cpp = glob_cols_no / num_processes; //podstawa
    int more = glob_cols_no % num_processes; //liczba procesow ktorym dodamy 1 (pierwsze z brzegu)

    int procmorecols = more * (cpp + 1); //liczba kolumn zabranych przez procesy z wieksza iloscia kolumn
    if (col_no <= procmorecols) { //indeks w kolumnie procesu z wieksza iloscia kolumn
        return col_no % (cpp + 1);
    } else { //indeks w kolumnie procesu z mniejsza iloscia kolumn
        return (col_no - procmorecols) % cpp;
    }
}

void split_sparse_to_broadcast_blocked(sparse_type *msgs, sparse_type *sparse, int num_processes) {
    int i, j;
    int *A_inc = malloc(sizeof(int) * num_processes);
    int *nnzs = malloc(sizeof(int) * num_processes);
    for (i = 0; i < num_processes; i++) {
        A_inc[i] = nnzs[i] = 0;
    }

    int nnz = sparse->IA[sparse->rows_no];
    for (i = 0; i < nnz; i++) {
        nnzs[col_to_proc(sparse->JA[i], num_processes, sparse->cols_no)]++;
    }

    for (i = 0; i < num_processes; i++) {
        alloc_sparse(msgs + i, nnzs[i], sparse->rows_no, proc_to_cols_no(i, num_processes, sparse->cols_no));
    }

    j = 1;
    while (j <= sparse->rows_no) {
        for (i = sparse->IA[j - 1]; i < sparse->IA[j]; i++) {
            int ind = col_to_proc(sparse->JA[i], num_processes, sparse->cols_no);
            msgs[ind].A[A_inc[ind]] = sparse->A[i];
            msgs[ind].JA[A_inc[ind]++] = col_to_col_part(sparse->JA[i], num_processes, sparse->cols_no);
        }
        for (i = 0; i < num_processes; i++) {
            msgs[i].IA[j] = A_inc[i];
        }
        ++j;
    }

    for (i = 0; i < num_processes; i++) {
        msgs[i].IA[0] = 0;
    }

    free(nnzs);
    free(A_inc);
}

void get_sparse_part_blocked(int mpi_rank, int num_processes, sparse_type *sparse, sparse_type *Apart) {
    if (mpi_rank == ROOT) {
        sparse_type *msgs = malloc(num_processes * sizeof(sparse_type));
        split_sparse_to_broadcast_blocked(msgs, sparse, num_processes);
        int i;
        for (i = 1; i < num_processes; i++) {
            send_sparse(i, msgs + i);
        }
        for (i = 1; i < num_processes; i++) {
            free_sparse(msgs + i);
        }

	Apart->IA = malloc((msgs->rows_no + 1) * sizeof(int));
	Apart->JA = malloc((msgs->IA[msgs->rows_no]) * sizeof(int));
	Apart->A = malloc((msgs->IA[msgs->rows_no]) * sizeof(double));
	Apart->A = msgs->A;
	Apart->JA = msgs->JA;
	Apart->IA = msgs->IA;
	Apart->cols_no = msgs->cols_no;
	Apart->rows_no = msgs->rows_no;
        free(msgs);
    } else {
    	receive_sparse(0, Apart);
    }
}

void broadcast_A_parts_in_groups(int mpi_rank, sparse_type *As, sparse_type *A, int sub_size, MPI_Comm *sub_comm) {
    int *displs = malloc(sub_size * sizeof(int));
    int *recvcounts = malloc(sub_size * sizeof(int));
    displs[0] = 0;

    int i;
    for (i = 0; i < sub_size; i++) {
        displs[i] = &(As[i].cols_no) - &(As[0].cols_no); //offsety w strukturze
        recvcounts[i] = 1;
    }

    MPI_Allgatherv(&(A->cols_no), 1, MPI_INT, &(As[0].cols_no), recvcounts, displs, MPI_INT, *sub_comm);

    for (i = 0; i < sub_size; i++) {
        As[i].rows_no = A->rows_no;
        As[i].IA = malloc((As[i].rows_no + 1) * sizeof(int));
        displs[i] = (As[i].IA - As[0].IA); //offsety w strukturze
        recvcounts[i] = As[i].rows_no + 1;
    }

    MPI_Allgatherv(A->IA, A->rows_no + 1, MPI_INT, As[0].IA, recvcounts, displs, MPI_INT, *sub_comm);

    for (i = 0; i < sub_size; i++) {
        As[i].rows_no = A->rows_no;
        recvcounts[i] = As[i].IA[As[i].rows_no];
        As[i].JA = malloc(recvcounts[i] * sizeof(int));
        As[i].A = malloc(recvcounts[i] * sizeof(double));
    }

    displs[0] = 0;
    for (i = 1; i < sub_size; i++) {
        displs[i] = (As[i].JA - As[0].JA); //offsety w strukturze
    }

    MPI_Allgatherv(A->JA, A->IA[A->rows_no], MPI_INT, As[0].JA, recvcounts, displs, MPI_INT, *sub_comm);

    displs[0] = 0;
    for (i = 1; i < sub_size; i++) {
        displs[i] = (As[i].A - As[0].A); //offsety w strukturze
    }

    MPI_Allgatherv(A->A, A->IA[A->rows_no], MPI_DOUBLE, As[0].A, recvcounts, displs, MPI_DOUBLE, *sub_comm);

    free(displs);
    free(recvcounts);
}

void join_sparse_type_blocked(int mpi_rank, int sub_size, int num_processes, sparse_type *As, sparse_type *A) {
    int nnz = 0;
    int i, j, k, c = 0, cols_no = 0;
    for (i = 0; i < sub_size; i++) {
        nnz += As[i].IA[As[i].rows_no];
        cols_no += As[i].cols_no;
    }
    A->rows_no = As[0].rows_no;
    A->cols_no = cols_no;

    A->IA = malloc((As[0].rows_no + 1) * sizeof(int));
    A->JA = malloc(nnz * sizeof(int));
    A->A = malloc(nnz * sizeof(double));
    A->IA[0] = 0;

    for (j = 1; j <= A->rows_no; ++j) {
        int cols_counter = 0;
        for (i = 0; i < sub_size; ++i) {
            for (k = As[i].IA[j - 1]; k < As[i].IA[j]; ++k) {
                A->A[c] = As[i].A[k];
                A->JA[c] = cols_counter + As[i].JA[k];
                ++c;
            }
            cols_counter += As[i].cols_no;
        }
        A->IA[j] = c;
    }
}

void calculate_single_round_blocked(int row_B_offset, int sub_size, int num_processes, int mpi_rank,
                            dense_type *C, sparse_type *A, dense_type *B) {
    int i, curr_row = 0, k;
    while (curr_row < A->rows_no) {
        for (i = A->IA[curr_row]; i < A->IA[curr_row + 1]; ++i) {
            for (k = 0; k < B->cols_no; ++k) {
                int row_B = (A->JA[i] + row_B_offset) % A->rows_no; //liczba wierszy = liczba kolumn
                C->vals[ARRAY_IND(curr_row, k, B->cols_no)] +=
                        A->A[i] * B->vals[ARRAY_IND(row_B, k, B->cols_no)];
            }
        }
        ++curr_row;
    }
}

int get_offset(int round, int mpi_rank, int sub_size, int num_processes, int all_cols_no, int rounds_no) {
    int sub_ind = mpi_rank / sub_size; //indeks grupy
    int sub_offset_ind = (sub_ind + round) % rounds_no; //indeks grupy z uwzlednieniem aktualnej rundy
    int rank = sub_size * sub_offset_ind;
    int res = col_part_to_col(0, rank, num_processes, all_cols_no); //+1 grupa z indeksem 0

    return res;
}

void compute_matrix_blocked(int exponent, int sub_size, int num_processes, int mpi_rank, int all_cols_no,
                    dense_type *C, sparse_type *As, dense_type *B) {
    alloc_dense(B->cols_no, B->rows_no, C);
    sparse_type *recv = malloc(sizeof(sparse_type));
    recv->rows_no = As->rows_no;
    recv->IA = malloc((As->rows_no + 1) * sizeof(int));

    while (exponent--) {
        int i = 0;
        for (i = 0; i < C->cols_no * C->rows_no; i++) {
            C->vals[i] = 0;
        }

        //
        MPI_Request req[4];
        int c = num_processes / sub_size;
        int destination = (mpi_rank + (c - 1) * sub_size) % num_processes;
        int source = (mpi_rank + sub_size) % num_processes;

        for (i = 0; i < c; i++) {
            send_sparse_async(req, destination, As);
            int row_B_offset = get_offset(i, mpi_rank, sub_size, num_processes, all_cols_no, c);
            calculate_single_round_blocked(row_B_offset, sub_size, num_processes, mpi_rank, C, As, B);
            receive_sparse_rows_no(source, As->rows_no, recv);
            MPI_Waitall(4, req, MPI_STATUS_IGNORE);
            sparse_type *tmp = recv;
            recv = As;
            As = tmp;
        }
        copy_dense(C, B);
    }

    //free
    free(recv->IA);
}

void gather_and_show_results(int mpi_rank, int num_processes, int all_cols_no, int rows_no, dense_type *part) {
    if (mpi_rank != 0) {
        MPI_Gatherv(&(part->cols_no), 1, MPI_INT, NULL, NULL,
                    NULL, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Gatherv(part->vals, rows_no * part->cols_no, MPI_DOUBLE, NULL, NULL,
                    NULL, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        return;
    }

    dense_type all[num_processes];
    int i;
    int displs[num_processes];
    int recvcounts[num_processes];
    for (i = 0; i < num_processes; ++i) {
        all[i].rows_no = rows_no;
    }

    for (i = 0; i < num_processes; ++i) {
        recvcounts[i] = 1;
        displs[i] = &(all[i].cols_no) - &(all[0].cols_no);
    }
    MPI_Gatherv(&(part->cols_no), 1, MPI_INT, &(all->cols_no), recvcounts,
                displs, MPI_INT, 0, MPI_COMM_WORLD);

    for (i = 0; i < num_processes; ++i) {
        recvcounts[i] = all[i].cols_no * all[i].rows_no;
        all[i].vals = malloc(all[i].cols_no * all[i].rows_no * sizeof(double));
        displs[i] = all[i].vals - all[0].vals;
    }
    MPI_Gatherv(part->vals, part->rows_no * part->cols_no, MPI_DOUBLE, all->vals, recvcounts,
                displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    printf("%d %d\n", all[0].rows_no, all[0].rows_no);
    int j, k;
    for (i = 0; i < all[0].rows_no; ++i) { //liczba wierszy taka sama wszedzie
        for (k = 0; k < num_processes; ++k) {
            for (j = 0; j < all[k].cols_no; ++j) {
                printf("%.6lf ", all[k].vals[ARRAY_IND(i, j, all[k].cols_no)]);
            }
        }
        printf("\n");
    }

    //free
    for (i = 0; i < num_processes; i++) {
        free(all[i].vals);
    }
}

void count_and_print_ge_elements(int mpi_rank, int num_processes, dense_type *C, double ge_elm) {
    int i, j, ge = 0;
    for (i = 0; i < C->rows_no; ++i) {
        for (j = 0; j < C->cols_no; ++j) {
            if (C->vals[ARRAY_IND(i, j, C->cols_no)] >= ge_elm) {
                ++ge;
            }
        }
    }

    if (mpi_rank != 0) {
        MPI_Gather(&ge, 1, MPI_INT, NULL, NULL, NULL, 0, MPI_COMM_WORLD);
        return;
    }

    int ges[num_processes];
    MPI_Gather(&ge, 1, MPI_INT, ges, 1, MPI_INT, 0, MPI_COMM_WORLD);

    for (i = 1; i < num_processes; ++i) {
        ge += ges[i];
    }

    printf("%d\n", ge);
}
