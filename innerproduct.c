#include "innerproduct.h"
#include "utils.h"

//dla danego procesu zwraca liczbe kolumn w podzielonym A
inline int proc_to_rows_no(int mpi_rank, int num_processes, int rows_no) {
    int cpp = rows_no/num_processes; //podstawa
    int more = rows_no % num_processes; //liczba procesow ktorym dodamy 1 (pierwsze z brzegu)

    return cpp + ((mpi_rank < more) ? 1 : 0);
}

void split_sparse_to_broadcast_inner(sparse_type *msgs, sparse_type *sparse, int num_processes) {
    int i, j;
    int rows_all = 0;
    for (i = 0; i < num_processes; i++) {
        int rows = proc_to_rows_no(i, num_processes, sparse->rows_no);
        rows_all += rows;
        alloc_sparse(msgs + i, sparse->IA[rows_all] - sparse->IA[rows_all-rows], rows, sparse->cols_no);

        for (j = sparse->IA[rows_all-rows]; j < sparse->IA[rows_all]; j++) {
            int ind = j - sparse->IA[rows_all-rows];
            msgs[i].A[ind] = sparse->A[j];
            msgs[i].JA[ind] = sparse->JA[j];
        }

        msgs[i].IA[0] = 0;
        for (j = rows_all - rows; j < rows_all; j++) {
            int ind = j - (rows_all - rows);
            msgs[i].IA[ind+1] = sparse->IA[j+1] - sparse->IA[j] + msgs[i].IA[ind];
        }
    }

    for (i = 0; i < num_processes; i++) {
        msgs[i].IA[0] = 0;
    }
}

void get_sparse_part_inner(int mpi_rank, int num_processes, sparse_type* sparse, sparse_type* Apart) {
    if (mpi_rank == ROOT) {
        sparse_type *msgs = malloc(num_processes * sizeof(sparse_type));
        split_sparse_to_broadcast_inner(msgs, sparse, num_processes);
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

void broadcast_A_B_parts_in_groups(int mpi_rank, sparse_type *As, sparse_type *A, dense_type *Bs, dense_type *B,
                                   int sub_size, MPI_Comm *sub_comm) {
    int *displs = malloc(sub_size * sizeof(int));
    int *recvcounts = malloc(sub_size * sizeof(int));
    displs[0] = 0;

    //wysylamy A
    int i;
    for (i = 0; i < sub_size; i++) {
        displs[i] = &(As[i].cols_no) - &(As[0].cols_no); //offsety w strukturze
        recvcounts[i] = 1;
    }
    MPI_Allgatherv(&(A->cols_no), 1, MPI_INT, &(As[0].cols_no), recvcounts, displs, MPI_INT, *sub_comm);

    for (i = 0; i < sub_size; i++) {
        displs[i] = &(As[i].rows_no) - &(As[0].rows_no); //offsety w strukturze
        recvcounts[i] = 1;
    }
    MPI_Allgatherv(&(A->rows_no), 1, MPI_INT, &(As[0].rows_no), recvcounts, displs, MPI_INT, *sub_comm);

    for (i = 0; i < sub_size; i++) {
        As[i].IA = malloc((As[i].rows_no + 1) * sizeof(int));
        displs[i] = (As[i].IA - As[0].IA); //offsety w strukturze
        recvcounts[i] = As[i].rows_no + 1;
    }
    MPI_Allgatherv(A->IA, A->rows_no + 1, MPI_INT, As[0].IA, recvcounts, displs, MPI_INT, *sub_comm);

    for (i = 0; i < sub_size; i++) {
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

    MPI_Barrier(MPI_COMM_WORLD);
    //wysylamy B
    for (i = 0; i < sub_size; i++) {
        displs[i] = &(Bs[i].cols_no) - &(Bs[0].cols_no); //offsety w strukturze
        recvcounts[i] = 1;
    }
    MPI_Allgatherv(&(B->cols_no), 1, MPI_INT, &(Bs[0].cols_no), recvcounts, displs, MPI_INT, *sub_comm);

    for (i = 0; i < sub_size; i++) {
        displs[i] = &(Bs[i].rows_no) - &(Bs[0].rows_no); //offsety w strukturze
        recvcounts[i] = 1;
    }
    MPI_Allgatherv(&(B->rows_no), 1, MPI_INT, &(Bs[0].rows_no), recvcounts, displs, MPI_INT, *sub_comm);

    for (i = 0; i < sub_size; i++) {
        recvcounts[i] = Bs[i].rows_no * Bs[i].cols_no;
        Bs[i].vals = malloc((Bs[i].rows_no * Bs[i].cols_no) * sizeof(double));
        displs[i] = (Bs[i].vals - Bs[0].vals); //offsety w strukturze
    }

    MPI_Allgatherv(B->vals, B->rows_no * B->cols_no, MPI_DOUBLE, Bs[0].vals, recvcounts, displs, MPI_DOUBLE, *sub_comm);
    free(displs);
    free(recvcounts);
}

void join_sparse_type_inner(int mpi_rank, int sub_size, int num_processes, sparse_type *As, sparse_type *A) {
    int nnz = 0;
    int i, j, k = 0, rows_no = 0;
    for (i = 0; i < sub_size; i++) {
        nnz += As[i].IA[As[i].rows_no];
        rows_no += As[i].rows_no;
    }
    A->rows_no = rows_no;
    A->cols_no = As[0].cols_no;

    A->IA = malloc((A->rows_no + 1) * sizeof(int));
    A->JA = malloc(nnz * sizeof(int));
    A->A = malloc(nnz * sizeof(double));
    A->IA[0] = 0;

    int last_col = -1;
    int l = 0;
    for (i = 0; i < sub_size; i++) {
        for (j = 0; j < As[i].IA[As[i].rows_no]; j++) {
            A->A[k] = As[i].A[j];
            A->JA[k] = As[i].JA[j];
            last_col = A->JA[k];
            k++;
        }
    }

    l = 1, k = 0;
    for (i = 0; i < sub_size; i++) {
        for (j = 1; j <= As[i].rows_no; j++) {
            A->IA[l] = As[i].IA[j] + k;
            l++;
        }
        k += As[i].IA[As[i].rows_no];
    }
}

void join_dense_type_inner(int mpi_rank, int sub_size, int num_processes, dense_type *Bs, dense_type *B) {
    int i, j, k, cols_no = 0;
    for (i = 0; i < sub_size; i++) {
        cols_no += Bs[i].cols_no;
    }

    B->cols_no = cols_no;
    B->rows_no = Bs[0].rows_no; //liczba wierszy taka sama w kazdym wezle
    B->vals = malloc(B->rows_no * B->cols_no * sizeof(double));

    cols_no = 0;
    for (i = 0; i < sub_size; i++) {
        for (j = 0; j < B->rows_no; j++) {
            for (k = 0; k < Bs[i].cols_no; k++) {
                B->vals[ARRAY_IND(j, k+cols_no, B->cols_no)] = Bs[i].vals[ARRAY_IND(j, k, Bs[i].cols_no)];
            }
        }
        cols_no += Bs[i].cols_no;
    }
}

void calculate_single_round_inner(int row_C_offset, int sub_size, int num_processes, int mpi_rank, int n,
                                  dense_type *C, sparse_type *A, dense_type *B) {
    int i, curr_row = 0, k;
    while (curr_row < A->rows_no) {
        for (i = A->IA[curr_row]; i < A->IA[curr_row + 1]; ++i) {
            for (k = 0; k < B->cols_no; ++k) {
                C->vals[ARRAY_IND((curr_row + row_C_offset) % n, k, B->cols_no)] +=
                        A->A[i] * B->vals[ARRAY_IND(A->JA[i], k, B->cols_no)];
            }
        }
        ++curr_row;
    }
}

int first_index(int mpi_rank, int n, int num_processes) {
    int ext = n % num_processes;
    return (n / num_processes) * mpi_rank + (ext < mpi_rank ? ext : mpi_rank);
}

void send_sparse_async_inner(MPI_Request *req, int desination, sparse_type *msg) {
    MPI_Isend(&(msg->cols_no), 1, MPI_INT, desination, COLS_NO_TAG, MPI_COMM_WORLD, req);
    MPI_Isend(&(msg->rows_no), 1, MPI_INT, desination, ROWS_NO_TAG, MPI_COMM_WORLD, req + 1);
    MPI_Isend(msg->IA, msg->rows_no + 1, MPI_INT, desination, IA_TAG, MPI_COMM_WORLD, req + 2);
    MPI_Isend(msg->JA, msg->IA[msg->rows_no], MPI_INT, desination, JA_TAG, MPI_COMM_WORLD, req + 3);
    MPI_Isend(msg->A, msg->IA[msg->rows_no], MPI_DOUBLE, desination, A_TAG, MPI_COMM_WORLD, req + 4);
}

void receive_sparse_rows_no_inner(int source, sparse_type *msg) {
    MPI_Recv(&(msg->cols_no), 1, MPI_DOUBLE, source, COLS_NO_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(&(msg->rows_no), 1, MPI_DOUBLE, source, ROWS_NO_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    msg->IA = malloc((msg->rows_no + 1) * sizeof(int));

    MPI_Recv(msg->IA, msg->rows_no + 1, MPI_INT, source, IA_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    int nnz = msg->IA[msg->rows_no];

    msg->JA = malloc(nnz * sizeof(int));
    msg->A = malloc(nnz * sizeof(double));

    MPI_Recv(msg->JA, nnz, MPI_INT, source, JA_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(msg->A, nnz, MPI_DOUBLE, source, A_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

void sendrecv_sparse(int mpi_rank, int source, int destination, sparse_type *A, sparse_type *recv) {
    int old_rows = A->rows_no;
    int old_nnz = A->IA[old_rows];

    MPI_Sendrecv(&(A->cols_no), 1, MPI_INT, destination, COLS_NO_TAG, &(recv->cols_no),
                1, MPI_INT, source, COLS_NO_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    MPI_Sendrecv(&(A->rows_no), 1, MPI_INT, destination, ROWS_NO_TAG, &(recv->rows_no),
                1, MPI_INT, source, ROWS_NO_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    recv->IA = malloc((recv->rows_no + 1) *  sizeof(int));
    MPI_Sendrecv(A->IA, old_rows + 1, MPI_INT, destination, IA_TAG, recv->IA,
                recv->rows_no + 1, MPI_INT, source, IA_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    recv->JA = malloc(recv->IA[recv->rows_no] *  sizeof(int));
    MPI_Sendrecv(A->JA, old_nnz, MPI_INT, destination, JA_TAG, recv->JA,
                recv->IA[recv->rows_no], MPI_INT, source, JA_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    recv->A = malloc(recv->IA[recv->rows_no] *  sizeof(double));
    MPI_Sendrecv(A->A, old_nnz, MPI_DOUBLE, destination, IA_TAG, recv->A,
                 recv->IA[recv->rows_no], MPI_DOUBLE, source, IA_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

void compute_matrix_inner(int exponent, int sub_size, int num_processes, int mpi_rank, int all_cols_no, MPI_Comm *sub_comm,
                            dense_type *C, sparse_type *A, dense_type *B) {
    alloc_dense(B->cols_no, B->rows_no, C);

    //init shift
    int q = num_processes / sub_size / sub_size;
    int layer = mpi_rank % sub_size;
    if (layer > 0) {
        sparse_type *recv = malloc(sizeof(sparse_type));
        int source = (mpi_rank + sub_size * layer * q) % num_processes;
        int destination = (mpi_rank - sub_size * layer * q + num_processes) % num_processes;
        sendrecv_sparse(mpi_rank, source, destination, A, recv);
        A = recv;
    }

    int col_idx = mpi_rank / sub_size;
    int destination = (mpi_rank - sub_size + num_processes) % num_processes;
    int source = (mpi_rank + sub_size) % num_processes;

    int row_C_offset = first_index(((layer * q + col_idx) * sub_size) % num_processes, all_cols_no, num_processes);

    sparse_type *recv = malloc(sizeof(sparse_type));
    while (exponent--) {
        int i = 0;
        for (i = 0; i < C->cols_no * C->rows_no; i++) {
            C->vals[i] = 0;
        }
        C->vals[0] = 0;

        //
        MPI_Request req[5];
        for (i = 0; i < q; i++) {
            send_sparse_async_inner(req, destination, A);
            calculate_single_round_inner(row_C_offset, sub_size, num_processes, mpi_rank, all_cols_no, C, A, B);
            row_C_offset += A->rows_no;
            receive_sparse_rows_no_inner(source, recv);
            MPI_Waitall(5, req, MPI_STATUS_IGNORE);
            sparse_type *tmp = recv;
            recv = A;
            A = tmp;
        }

        int size = C->rows_no * C->cols_no;
        double *buff = malloc(size * sizeof(double));
        MPI_Allreduce(C->vals, buff, size, MPI_DOUBLE, MPI_SUM, *sub_comm);
        for (i = 0; i < size; i++) {
            C->vals[i] = buff[i];
        }

        //free
        free(buff);

        copy_dense(C, B);
    }
}

void gather_and_show_results_inner(int mpi_rank, int sub_size, int num_processes, dense_type *C) {
    MPI_Comm sub_comm;

    int color = mpi_rank % sub_size;
    MPI_Comm_split(MPI_COMM_WORLD, color, num_processes, &sub_comm);

    if (mpi_rank % sub_size != 0) { //bierzemy tylko z pierwszej warstwy
        MPI_Comm_free(&sub_comm);
        return;
    }

    if (mpi_rank != 0) {
        MPI_Gatherv(&(C->cols_no), 1, MPI_INT, NULL, NULL,
                    NULL, MPI_INT, 0, sub_comm);
        MPI_Gatherv(C->vals, C->rows_no * C->cols_no, MPI_DOUBLE, NULL, NULL,
                    NULL, MPI_DOUBLE, 0, sub_comm);
        MPI_Comm_free(&sub_comm);
        return;
    }

    int np = num_processes / sub_size;
    dense_type all[np];
    int i;
    int displs[np];
    int recvcounts[np];
    for (i = 0; i < np; ++i) {
        all[i].rows_no = C->rows_no;
    }

    for (i = 0; i < np; ++i) {
        recvcounts[i] = 1;
        displs[i] = &(all[i].cols_no) - &(all[0].cols_no);
    }

    MPI_Gatherv(&(C->cols_no), 1, MPI_INT, &(all->cols_no), recvcounts,
                displs, MPI_INT, 0, sub_comm);

    for (i = 0; i < np; ++i) {
        recvcounts[i] = all[i].cols_no * all[i].rows_no;
        all[i].vals = malloc(all[i].cols_no * all[i].rows_no * sizeof(double));
        displs[i] = all[i].vals - all[0].vals;
    }
    MPI_Gatherv(C->vals, C->rows_no * C->cols_no, MPI_DOUBLE, all->vals, recvcounts,
                displs, MPI_DOUBLE, 0, sub_comm);

    printf("%d %d\n", all[0].rows_no, all[0].rows_no);
    int j, k;
    for (i = 0; i < all[0].rows_no; ++i) { //liczba wierszy taka sama wszedzie
        for (k = 0; k < np; ++k) {
            for (j = 0; j < all[k].cols_no; ++j) {
                printf("%.6lf ", all[k].vals[ARRAY_IND(i, j, all[k].cols_no)]);
            }
        }
        printf("\n");
    }

    MPI_Comm_free(&sub_comm);
    //free
    for (i = 0; i < np; i++) {
        free(all[i].vals);
    }
}

void count_and_print_ge_elements_inner(int mpi_rank, int num_processes, int sub_size, dense_type *C, double ge_elm) {
    MPI_Comm sub_comm;
    int color = mpi_rank % sub_size;
    MPI_Comm_split(MPI_COMM_WORLD, color, num_processes, &sub_comm);

    int np = num_processes / sub_size;
    int i, j, ge = 0;
    for (i = 0; i < C->rows_no; ++i) {
        for (j = 0; j < C->cols_no; ++j) {
            if (C->vals[ARRAY_IND(i, j, C->cols_no)] >= ge_elm) {
                ++ge;
            }
        }
    }

    if (mpi_rank % sub_size != 0) {
        MPI_Comm_free(&sub_comm);
        return;
    }

    if (mpi_rank != 0) {
        MPI_Gather(&ge, 1, MPI_INT, NULL, NULL, NULL, 0, sub_comm);
        MPI_Comm_free(&sub_comm);
        return;
    }

    int ges[np];
    MPI_Gather(&ge, 1, MPI_INT, ges, 1, MPI_INT, 0, sub_comm);

    for (i = 1; i < np; ++i) {
        ge += ges[i];
    }

    printf("%d\n", ge);
    MPI_Comm_free(&sub_comm);
}
